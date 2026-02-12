"""
Refactored HDF5 dataset preparator with improved code structure and readability.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split

from fdq.ui_functions import startProgBar, iprint


class DatasetType(Enum):
    """Enum for dataset types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    is_3d: bool = False
    file_extension: str = "hdf"
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    subset_train: float = 1.0
    subset_val: float = 1.0
    subset_test: float = 1.0
    val_from_train_ratio: float = 0.0


@dataclass
class SampleInfo:
    """Information about a sample in the dataset."""
    file_idx: int
    group_idx: int
    dataset_idx: int
    sample_idx: int


class TensorFormatter:
    """Utility class for tensor formatting operations."""
    
    @staticmethod
    def to_BCHW(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Reshape tensor to BCHW format (Batch, Channel, Height, Width).
        Adds dimensions if batch or channel is missing.
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(np.array(tensor))
        
        if len(tensor.shape) == 3:
            if tensor.shape[0] > 3:
                # assume BHW format
                tensor = tensor.unsqueeze(1)
            else:
                # assume CHW format
                tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 2:
            # assume HW format
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    @staticmethod
    def to_BCDHW(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Reshape tensor to BCDHW format (Batch, Channel, Depth, Height, Width).
        Adds dimensions if batch or channel is missing.
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(np.array(tensor))
        
        if len(tensor.shape) == 3:
            # assume DHW format
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError(
                f"Unsupported tensor shape for 3D conversion: {tensor.shape}"
            )
        
        return tensor


class HDFKeyProcessor:
    """Utility class for processing HDF key requests."""
    
    @staticmethod
    def cleanup_hdf_keys(requests: Optional[List[Union[str, Dict[str, str]]]]) -> Optional[Tuple[List[str], List[str]]]:
        """
        Clean up and validate HDF key requests.
        
        Args:
            requests: List of key requests, either strings or dictionaries
            
        Returns:
            Tuple of (keys, transform_names) or None if no requests
        """
        if requests is None:
            return None, None
        
        if not isinstance(requests, list):
            raise ValueError("hdf_ds_key_request must be a list of strings or dictionaries")
        
        if not requests:
            return None, None
        
        if isinstance(requests[0], dict):
            # Handle dictionary format: [{"key": "transform_name"}]
            keys = []
            transform_names = []
            for req_dict in requests:
                if len(req_dict) != 1:
                    raise ValueError("Each dictionary in hdf_ds_key_request must have exactly one key-value pair")
                key, transform_name = next(iter(req_dict.items()))
                keys.append(key)
                transform_names.append(transform_name)
            return keys, transform_names
        else:
            # Handle string format: ["key1", "key2"]
            return requests, None


class HDF5FileProcessor:
    """Handles processing of individual HDF5 files."""
    
    def __init__(self, config: DatasetConfig, formatter: TensorFormatter):
        self.config = config
        self.formatter = formatter
    
    def analyze_file_structure(self, h5_file: h5py.File, key_requests: Optional[List[str]]) -> Tuple[List[str], bool, bool]:
        """
        Analyze the structure of an HDF5 file.
        
        Returns:
            Tuple of (group_names, has_groups, datasets_are_sample_pairs, groups_are_sample_pairs)
        """
        root_keys = list(h5_file.keys())
        has_key_requests = key_requests is not None and len(key_requests) > 0
        
        # Check if file has groups
        has_groups = isinstance(h5_file[root_keys[0]], h5py._hl.group.Group)
        
        if has_groups:
            group_names = root_keys
        else:
            group_names = ["ROOTGROUP"]
        
        # Check if datasets are sample pairs (all requested keys are in root)
        datasets_are_sample_pairs = (
            has_key_requests and 
            not has_groups and 
            all(key in root_keys for key in key_requests)
        )
        
        # Check if groups are sample pairs (all requested keys are group names)
        groups_are_sample_pairs = (
            has_key_requests and 
            has_groups and 
            all(key in group_names for key in key_requests)
        )
        
        return group_names, datasets_are_sample_pairs, groups_are_sample_pairs
    
    def process_group_samples(self, h5_file: h5py.File, key_requests: List[str], 
                            file_path: Path, show_progress: bool = False) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
        """Process samples when groups are sample pairs."""
        tensor_per_group = []
        samples_per_group = []
        
        for group_key in key_requests:
            if group_key not in h5_file:
                raise ValueError(f"Group '{group_key}' not found in {file_path}")
            
            group_datasets = list(h5_file[group_key].keys())
            tensor_per_dataset = []
            
            pbar = startProgBar(
                len(group_datasets),
                f"Processing {file_path.name}, Group: {group_key} ({len(group_datasets)} datasets)",
                is_active=show_progress,
            )
            
            for i, dataset_key in enumerate(group_datasets):
                pbar.update(i)
                dataset = h5_file[group_key][dataset_key]
                
                if self.config.is_3d:
                    tensor = self.formatter.to_BCDHW(dataset)
                else:
                    tensor = self.formatter.to_BCHW(dataset)
                
                tensor_per_dataset.append(tensor)
            
            pbar.finish()
            
            tensor_per_group.append(tensor_per_dataset)
            samples_per_group.append([t.shape[0] for t in tensor_per_dataset])
        
        # Validate that all groups have the same number of samples
        if not all(counts == samples_per_group[0] for counts in samples_per_group):
            raise ValueError(f"Number of samples per group in {file_path} does not match")
        
        return tensor_per_group, [samples_per_group[0]]
    
    def process_dataset_samples(self, h5_file: h5py.File, key_requests: List[str], 
                              file_path: Path, show_progress: bool = False) -> Tuple[List[List[torch.Tensor]], int]:
        """Process samples when datasets are sample pairs."""
        if self.config.is_3d:
            raise NotImplementedError("3D data not supported for dataset sample pairs")
        
        tensor_per_dataset = []
        
        pbar = startProgBar(
            len(key_requests),
            f"Processing {file_path.name} ({len(key_requests)} datasets)",
            is_active=show_progress,
        )
        
        for i, dataset_key in enumerate(key_requests):
            pbar.update(i)
            
            if dataset_key not in h5_file:
                raise ValueError(f"Dataset '{dataset_key}' not found in {file_path}")
            
            tensor = self.formatter.to_BCHW(h5_file[dataset_key])
            tensor_per_dataset.append(tensor)
        
        pbar.finish()
        
        # Validate that all datasets have the same number of samples
        sample_counts = [t.shape[0] for t in tensor_per_dataset]
        if not all(count == sample_counts[0] for count in sample_counts):
            raise ValueError(f"Number of samples per dataset in {file_path} does not match")
        
        return [tensor_per_dataset], sample_counts[0]
    
    def process_regular_samples(self, h5_file: h5py.File, group_names: List[str], 
                              key_requests: Optional[List[str]], file_path: Path, 
                              show_progress: bool = False) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
        """Process regular samples (not pairs)."""
        if self.config.is_3d:
            raise NotImplementedError("3D data not supported for regular samples")
        
        tensor_per_group = []
        samples_per_group = []
        
        for group_name in group_names:
            samples_per_dataset = []
            
            if group_name == "ROOTGROUP":
                # Process root-level datasets
                root_keys = list(h5_file.keys())
                tensor_per_dataset = []
                
                pbar = startProgBar(
                    len(root_keys),
                    f"Processing {file_path.name}",
                    is_active=show_progress,
                )
                
                for i, dataset_key in enumerate(root_keys):
                    pbar.update(i)
                    tensor = self.formatter.to_BCHW(h5_file[dataset_key])
                    samples_per_dataset.append(tensor.shape[0])
                    tensor_per_dataset.append(tensor)
                
                pbar.finish()
                tensor_per_group.append(tensor_per_dataset)
            else:
                # Process group-level datasets
                group_keys = list(h5_file[group_name].keys())
                
                if key_requests is None:
                    raise ValueError("Key requests required for group-level processing")
                
                # Validate requested keys exist in group
                missing_keys = [key for key in key_requests if key not in group_keys]
                if missing_keys:
                    raise ValueError(f"Keys {missing_keys} not found in group '{group_name}' of {file_path}")
                
                tensor_per_dataset = []
                sample_counts = []
                
                pbar = startProgBar(
                    len(key_requests),
                    f"Processing {file_path.name}",
                    is_active=show_progress,
                )
                
                for i, dataset_key in enumerate(key_requests):
                    pbar.update(i)
                    tensor = self.formatter.to_BCHW(h5_file[group_name][dataset_key])
                    sample_counts.append(tensor.shape[0])
                    tensor_per_dataset.append(tensor)
                
                pbar.finish()
                
                # Validate consistent sample counts
                if not all(count == sample_counts[0] for count in sample_counts):
                    raise ValueError(f"Inconsistent sample counts in group '{group_name}' of {file_path}")
                
                tensor_per_group.append(tensor_per_dataset)
                samples_per_dataset.append(sample_counts[0])
            
            samples_per_group.append(samples_per_dataset)
        
        return tensor_per_group, samples_per_group


class H5Dataset(data.Dataset):
    """
    Refactored HDF5 dataset loader with improved structure and readability.
    
    This class handles loading data from HDF5 files with various organizational structures:
    - Files with groups containing datasets
    - Files with datasets at root level
    - Sample pairs (where one sample consists of multiple datasets/groups)
    """
    
    def __init__(self, file_paths: List[Path], experiment: Any, 
                 hdf_ds_def: Optional[List[Union[str, Dict[str, str]]]] = None, 
                 config: Optional[DatasetConfig] = None):
        """
        Initialize the HDF5 dataset loader.
        
        Args:
            file_paths: List of paths to HDF5 files
            experiment: Experiment object containing transformers
            hdf_ds_def: Definition of datasets to load
            config: Dataset configuration
        """
        super().__init__()
        
        self.config = config or DatasetConfig()
        self.file_paths = file_paths
        self.filenames = [fp.stem for fp in file_paths]
        self.transformers = experiment.transformers
        
        # Process HDF key requests
        self.key_requests, self.transform_names = HDFKeyProcessor.cleanup_hdf_keys(hdf_ds_def)
        
        # Initialize components
        self.formatter = TensorFormatter()
        self.file_processor = HDF5FileProcessor(self.config, self.formatter)
        
        # Data storage
        self.data_tensors: List[List[List[torch.Tensor]]] = []  # [file][group][dataset]
        self.sample_counts: torch.Tensor = None  # [file][group][dataset]
        
        # Structure flags
        self.groups_are_sample_pairs: Optional[bool] = None
        self.datasets_are_sample_pairs: Optional[bool] = None
        self.group_names: Optional[List[str]] = None
        
        # Load and process files
        self._load_files()
        self._finalize_sample_counts()
    
    def _load_files(self) -> None:
        """Load and process all HDF5 files."""
        file_sample_counts = []
        
        show_file_progress = len(self.file_paths) > 1
        pbar = startProgBar(
            len(self.file_paths),
            f"Loading {len(self.file_paths)} HDF5 files",
            is_active=show_file_progress,
        )
        
        for file_idx, file_path in enumerate(self.file_paths):
            pbar.update(file_idx)
            
            try:
                with h5py.File(file_path, 'r') as h5_file:
                    file_tensors, file_counts = self._process_file(h5_file, file_path, not show_file_progress)
                    self.data_tensors.append(file_tensors)
                    file_sample_counts.append(file_counts)
            except Exception as e:
                raise RuntimeError(f"Error processing file {file_path}: {e}") from e
        
        pbar.finish()
        self.sample_counts = torch.tensor(file_sample_counts)
    
    def _process_file(self, h5_file: h5py.File, file_path: Path, show_progress: bool) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
        """Process a single HDF5 file."""
        # Analyze file structure
        group_names, datasets_are_pairs, groups_are_pairs = self.file_processor.analyze_file_structure(
            h5_file, self.key_requests
        )
        
        # Validate structure consistency across files
        self._validate_structure_consistency(group_names, datasets_are_pairs, groups_are_pairs)
        
        # Process based on structure type
        if self.groups_are_sample_pairs:
            return self.file_processor.process_group_samples(
                h5_file, self.key_requests, file_path, show_progress
            )
        elif self.datasets_are_sample_pairs:
            tensors, count = self.file_processor.process_dataset_samples(
                h5_file, self.key_requests, file_path, show_progress
            )
            return tensors, [[count]]
        else:
            return self.file_processor.process_regular_samples(
                h5_file, group_names, self.key_requests, file_path, show_progress
            )
    
    def _validate_structure_consistency(self, group_names: List[str], 
                                      datasets_are_pairs: bool, groups_are_pairs: bool) -> None:
        """Validate that all files have consistent structure."""
        if self.group_names is None:
            self.group_names = group_names
        elif self.group_names != group_names:
            raise ValueError("All files must have the same group structure")
        
        if self.datasets_are_sample_pairs is None:
            self.datasets_are_sample_pairs = datasets_are_pairs
        elif self.datasets_are_sample_pairs != datasets_are_pairs:
            raise ValueError("Dataset pair structure must be consistent across files")
        
        if self.groups_are_sample_pairs is None:
            self.groups_are_sample_pairs = groups_are_pairs
        elif self.groups_are_sample_pairs != groups_are_pairs:
            raise ValueError("Group pair structure must be consistent across files")
    
    def _finalize_sample_counts(self) -> None:
        """Finalize sample count tensor and compute total samples."""
        # Pad sample counts to ensure consistent tensor shape
        max_groups = max(len(file_counts) for file_counts in self.sample_counts)
        max_datasets = max(len(group_counts) for file_counts in self.sample_counts 
                          for group_counts in file_counts)
        
        padded_counts = []
        for file_counts in self.sample_counts:
            padded_file = []
            for group_counts in file_counts:
                padded_group = group_counts + [0] * (max_datasets - len(group_counts))
                padded_file.append(padded_group)
            # Pad groups if necessary
            while len(padded_file) < max_groups:
                padded_file.append([0] * max_datasets)
            padded_counts.append(padded_file)
        
        self.sample_counts = torch.tensor(padded_counts)
        self.total_samples = int(torch.sum(self.sample_counts))
    
    def _find_sample_location(self, index: int) -> SampleInfo:
        """Find the file, group, dataset, and sample index for a given global index."""
        # Find file
        file_idx = 0
        cumulative_samples = 0
        
        for file_idx in range(len(self.file_paths)):
            file_total = int(torch.sum(self.sample_counts[file_idx]))
            if index < cumulative_samples + file_total:
                break
            cumulative_samples += file_total
        
        # Adjust index to be relative to the file
        file_relative_index = index - cumulative_samples
        
        # Find group and dataset
        group_idx = 0
        dataset_idx = 0
        sample_idx = 0
        
        cumulative_in_file = 0
        for group_idx in range(self.sample_counts.shape[1]):
            for dataset_idx in range(self.sample_counts.shape[2]):
                dataset_samples = int(self.sample_counts[file_idx, group_idx, dataset_idx])
                if dataset_samples == 0:
                    continue
                
                if file_relative_index < cumulative_in_file + dataset_samples:
                    sample_idx = file_relative_index - cumulative_in_file
                    return SampleInfo(file_idx, group_idx, dataset_idx, sample_idx)
                
                cumulative_in_file += dataset_samples
        
        raise IndexError(f"Index {index} out of range")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Get a sample by index."""
        sample_info = self._find_sample_location(index)
        
        if self.groups_are_sample_pairs:
            return self._get_group_sample(sample_info)
        elif self.datasets_are_sample_pairs:
            return self._get_dataset_sample(sample_info)
        else:
            return self._get_regular_sample(sample_info)
    
    def _get_group_sample(self, sample_info: SampleInfo) -> Tuple[torch.Tensor, ...]:
        """Get a sample when groups are sample pairs."""
        samples = []
        
        for group_idx, group_key in enumerate(self.key_requests):
            tensor = self.data_tensors[sample_info.file_idx][group_idx][sample_info.dataset_idx][sample_info.sample_idx]
            
            if self.transform_names and group_idx < len(self.transform_names):
                transform_name = self.transform_names[group_idx]
                transform = self.transformers.get(transform_name)
                if transform:
                    tensor = transform(tensor)
            
            samples.append(tensor)
        
        return tuple(samples)
    
    def _get_dataset_sample(self, sample_info: SampleInfo) -> Tuple[torch.Tensor, ...]:
        """Get a sample when datasets are sample pairs."""
        samples = []
        
        for dataset_idx in range(len(self.key_requests)):
            tensor = self.data_tensors[sample_info.file_idx][0][dataset_idx][sample_info.sample_idx]
            
            if self.transform_names and dataset_idx < len(self.transform_names):
                transform_name = self.transform_names[dataset_idx]
                transform = self.transformers.get(transform_name)
                if transform:
                    tensor = transform(tensor)
            
            samples.append(tensor)
        
        return tuple(samples)
    
    def _get_regular_sample(self, sample_info: SampleInfo) -> Tuple[torch.Tensor, ...]:
        """Get a regular sample (not pairs)."""
        tensor = self.data_tensors[sample_info.file_idx][sample_info.group_idx][sample_info.dataset_idx]
        
        if tensor.dim() > 1:
            tensor = tensor[sample_info.sample_idx]
        
        # Apply transformation if specified
        if (self.transform_names and 
            self.key_requests and 
            len(self.transform_names) == 1 and 
            len(self.key_requests) == 1):
            transform_name = self.transform_names[0]
            transform = self.transformers.get(transform_name)
            if transform:
                tensor = transform(tensor.unsqueeze(0))
                return tuple(tensor)
        
        return (tensor.unsqueeze(0),)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.total_samples


class DatasetFactory:
    """Factory class for creating datasets and data loaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def create_datasets(self, experiment: Any, args: Any) -> Dict[str, Any]:
        """
        Create train, validation, and test datasets with data loaders.
        
        Returns:
            Dictionary containing data loaders and metadata
        """
        # Extract configuration from args
        self._update_config_from_args(args)
        
        # Get file paths
        data_files = self._get_data_files(args)
        
        # Create datasets
        datasets = self._create_raw_datasets(data_files, experiment, args)
        
        # Apply subsampling
        subsampled_datasets = self._apply_subsampling(datasets, args)
        
        # Create data loaders
        data_loaders = self._create_data_loaders(subsampled_datasets)
        
        # Return comprehensive results
        return self._create_results_dict(data_loaders, subsampled_datasets)
    
    def _update_config_from_args(self, args: Any) -> None:
        """Update configuration from arguments."""
        for attr in ['is_3d', 'train_batch_size', 'val_batch_size', 'test_batch_size',
                     'num_workers', 'pin_memory', 'drop_last', 'shuffle_train',
                     'shuffle_val', 'shuffle_test', 'subset_train', 'subset_val',
                     'subset_test', 'val_from_train_ratio']:
            if hasattr(args, attr):
                setattr(self.config, attr, getattr(args, attr))
    
    def _get_data_files(self, args: Any) -> Dict[DatasetType, List[Path]]:
        """Get data file paths based on configuration."""
        data_files = {}
        file_extension = getattr(args, 'file_extension', self.config.file_extension)
        
        if args.base_path is not None:
            # Search for files in base path and subdirectories
            for dataset_type in DatasetType:
                if dataset_type == DatasetType.ALL:
                    data_path = Path(args.base_path)
                else:
                    data_path = Path(args.base_path) / dataset_type.value
                
                if data_path.is_dir():
                    data_files[dataset_type] = sorted(data_path.glob(f"**/*.{file_extension}"))
                else:
                    data_files[dataset_type] = []
            
            # Use all files for training if no specific train set
            if not data_files[DatasetType.TRAIN] and data_files[DatasetType.ALL]:
                data_files[DatasetType.TRAIN] = data_files[DatasetType.ALL]
        
        else:
            # Use explicitly provided file paths
            data_files[DatasetType.TRAIN] = [Path(p) for p in (args.train_files_path or [])]
            data_files[DatasetType.VAL] = [Path(p) for p in (args.val_files_path or [])]
            data_files[DatasetType.TEST] = [Path(p) for p in (args.test_files_path or [])]
        
        return data_files
    
    def _create_raw_datasets(self, data_files: Dict[DatasetType, List[Path]], 
                           experiment: Any, args: Any) -> Dict[str, H5Dataset]:
        """Create raw datasets from file paths."""
        datasets = {}
        
        is_training = getattr(experiment.mode.op_mode, 'train', False)
        is_testing = getattr(experiment.mode.op_mode, 'test', False)
        
        if is_training and data_files[DatasetType.TRAIN]:
            iprint("Creating training dataset")
            datasets['train'] = H5Dataset(
                data_files[DatasetType.TRAIN], experiment, 
                args.hdf_ds_key_request, self.config
            )
            
            if data_files[DatasetType.VAL]:
                iprint("Creating validation dataset")
                datasets['val'] = H5Dataset(
                    data_files[DatasetType.VAL], experiment, 
                    args.hdf_ds_key_request, self.config
                )
        
        if is_testing and data_files[DatasetType.TEST]:
            iprint("Creating test dataset")
            test_key_request = getattr(args, 'hdf_ds_key_request_test', None) or args.hdf_ds_key_request
            datasets['test'] = H5Dataset(
                data_files[DatasetType.TEST], experiment, 
                test_key_request, self.config
            )
        
        return datasets
    
    def _apply_subsampling(self, datasets: Dict[str, H5Dataset], args: Any) -> Dict[str, data.Dataset]:
        """Apply subsampling to datasets."""
        subsampled = {}
        
        for name, dataset in datasets.items():
            if name == 'train':
                ratio = self.config.subset_train
                shuffle = self.config.shuffle_train
            elif name == 'val':
                ratio = self.config.subset_val
                shuffle = self.config.shuffle_train  # Use train shuffle setting
            else:  # test
                ratio = self.config.subset_test
                shuffle = self.config.shuffle_test
            
            n_samples = len(dataset)
            n_sub_samples = int(n_samples * ratio)
            
            if n_sub_samples == n_samples:
                subsampled[name] = dataset
            elif shuffle:
                subsampled[name], _ = random_split(dataset, [n_sub_samples, n_samples - n_sub_samples])
            else:
                subsampled[name] = data.Subset(dataset, range(n_sub_samples))
        
        # Handle train/val split if no validation set provided
        if 'train' in subsampled and 'val' not in subsampled and self.config.val_from_train_ratio > 0:
            self._split_train_val(subsampled, args)
        
        return subsampled
    
    def _split_train_val(self, datasets: Dict[str, data.Dataset], args: Any) -> None:
        """Split training set into train and validation."""
        train_dataset = datasets['train']
        n_train = len(train_dataset)
        n_val = int(n_train * self.config.val_from_train_ratio)
        n_train_new = n_train - n_val
        
        if self.config.shuffle_train:
            datasets['train'], datasets['val'] = random_split(train_dataset, [n_train_new, n_val])
        else:
            datasets['train'] = data.Subset(train_dataset, range(n_train_new))
            datasets['val'] = data.Subset(train_dataset, range(n_train_new, n_train))
    
    def _create_data_loaders(self, datasets: Dict[str, data.Dataset]) -> Dict[str, data.DataLoader]:
        """Create data loaders from datasets."""
        loaders = {}
        
        loader_configs = {
            'train': {
                'batch_size': self.config.train_batch_size,
                'shuffle': self.config.shuffle_train,
            },
            'val': {
                'batch_size': self.config.val_batch_size,
                'shuffle': self.config.shuffle_val,
            },
            'test': {
                'batch_size': self.config.test_batch_size,
                'shuffle': self.config.shuffle_test,
            }
        }
        
        for name, dataset in datasets.items():
            if len(dataset) > 0:
                config = loader_configs[name]
                loaders[name] = data.DataLoader(
                    dataset,
                    batch_size=config['batch_size'],
                    shuffle=config['shuffle'],
                    pin_memory=self.config.pin_memory,
                    num_workers=self.config.num_workers,
                    drop_last=self.config.drop_last,
                )
        
        return loaders
    
    def _create_results_dict(self, loaders: Dict[str, data.DataLoader], 
                           datasets: Dict[str, data.Dataset]) -> Dict[str, Any]:
        """Create comprehensive results dictionary."""
        return {
            'train_data_loader': loaders.get('train'),
            'val_data_loader': loaders.get('val'),
            'test_data_loader': loaders.get('test'),
            'n_train_samples': len(datasets.get('train', [])),
            'n_val_samples': len(datasets.get('val', [])),
            'n_test_samples': len(datasets.get('test', [])),
            'n_train_batches': len(loaders['train']) if 'train' in loaders else 0,
            'n_val_batches': len(loaders['val']) if 'val' in loaders else 0,
            'n_test_batches': len(loaders['test']) if 'test' in loaders else 0,
        }


def create_datasets(experiment: Any, args: Any = None) -> Dict[str, Any]:
    """
    Main entry point for creating datasets.
    
    This function maintains backward compatibility with the original interface
    while using the refactored implementation.
    """
    config = DatasetConfig()
    
    # Set data_is_3d from args if available
    if args and hasattr(args, 'data_is_3d'):
        config.is_3d = args.data_is_3d
    
    factory = DatasetFactory(config)
    return factory.create_datasets(experiment, args)
