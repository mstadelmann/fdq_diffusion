from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split

from fdq.ui_functions import startProgBar, iprint


class EmptyDataset(data.Dataset):
    """Empty dataset for cases where no data is available."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError("Empty dataset has no items")


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


def ensure_tensor_format(
    tensor: Union[np.ndarray, torch.Tensor], is_3d: bool = False
) -> torch.Tensor:
    """Convert to tensor and ensure proper format (BCHW for 2D, BCDHW for 3D)."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(np.array(tensor))

    if is_3d:
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(0).unsqueeze(0)  # [D,H,W] -> [B,C,D,H,W]
        else:
            raise ValueError(f"Unsupported 3D tensor shape: {tensor.shape}")
    else:
        if len(tensor.shape) == 2:
            return tensor.unsqueeze(0).unsqueeze(0)  # [H,W] -> [B,C,H,W]
        elif len(tensor.shape) == 3:
            if tensor.shape[0] > 3:
                return tensor.unsqueeze(1)  # [B,H,W] -> [B,C,H,W]
            else:
                return tensor.unsqueeze(0)  # [C,H,W] -> [B,C,H,W]
    return tensor


def parse_hdf_keys(
    requests: Optional[List],
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Parse HDF key requests into keys and transform names."""
    if requests is None:
        return None, None

    if not isinstance(requests, list):
        raise ValueError("hdf_ds_key_request must be a list!")

    if isinstance(requests[0], dict):
        if len(requests[0]) != 1:
            raise ValueError(
                "Each dict in hdf_ds_key_request must have exactly one key!"
            )
        keys, values = zip(*[(k, v) for d in requests for k, v in d.items()])
        return list(keys), list(values)
    else:
        raise NotImplementedError("Only dict format supported for hdf_ds_key_request")


class H5Dataset(data.Dataset):
    """Simplified HDF5 dataset loader."""

    def __init__(
        self,
        file_paths: List[Path],
        experiment,
        hdf_ds_def: Optional[List] = None,
        args=None,
    ):
        super().__init__()

        self.file_paths = file_paths
        self.is_3d = args.data_is_3d
        self.ds_keys, self.transform_names = parse_hdf_keys(hdf_ds_def)
        self.transformers = experiment.transformers

        # Store data: list of (file_idx, group_name, dataset_name, sample_idx) tuples
        self.sample_index = []
        self.data_cache = {}  # Cache loaded tensors

        self._load_data_structure()

    def _load_data_structure(self):
        """Build index of all samples across files."""
        pbar = startProgBar(len(self.file_paths), "Loading HDF5 structure")

        for file_idx, file_path in enumerate(self.file_paths):
            pbar.update(file_idx)

            with h5py.File(file_path, "r") as h5_file:
                self._index_file(h5_file, file_idx, file_path)

        pbar.finish()

    def _index_file(self, h5_file: h5py.File, file_idx: int, file_path: Path):
        """Index all samples in a single file."""
        root_keys = list(h5_file.keys())

        # Check if we have groups or datasets at root level
        has_groups = isinstance(h5_file[root_keys[0]], h5py._hl.group.Group)

        if has_groups:
            self._index_groups(h5_file, file_idx, file_path, root_keys)
        else:
            self._index_root_datasets(h5_file, file_idx, file_path, root_keys)

    def _index_groups(
        self, h5_file: h5py.File, file_idx: int, file_path: Path, group_names: List[str]
    ):
        """Index samples when data is organized in groups."""
        if not self.ds_keys:
            raise ValueError("Dataset keys must be specified for grouped data")

        # Check if keys match group names (group-based pairing)
        if all(key in group_names for key in self.ds_keys):
            self._index_group_pairs(h5_file, file_idx, file_path)
        else:
            # Keys are dataset names within groups
            self._index_datasets_in_groups(h5_file, file_idx, file_path, group_names)

    def _index_group_pairs(self, h5_file: h5py.File, file_idx: int, file_path: Path):
        """Index when each group contains matching datasets (group pairing)."""
        first_group = h5_file[self.ds_keys[0]]
        dataset_names = list(first_group.keys())

        for dataset_name in dataset_names:
            # Verify all groups have this dataset
            for group_key in self.ds_keys:
                if dataset_name not in h5_file[group_key]:
                    raise ValueError(
                        f"Dataset {dataset_name} not found in group {group_key}"
                    )

            # Get sample count from first group
            sample_count = h5_file[self.ds_keys[0]][dataset_name].shape[0]

            # Verify all groups have same sample count
            for group_key in self.ds_keys:
                if h5_file[group_key][dataset_name].shape[0] != sample_count:
                    raise ValueError(f"Sample count mismatch in {file_path}")

            # Add all samples from this dataset
            for sample_idx in range(sample_count):
                self.sample_index.append(
                    (file_idx, "GROUP_PAIR", dataset_name, sample_idx)
                )

    def _index_datasets_in_groups(
        self, h5_file: h5py.File, file_idx: int, file_path: Path, group_names: List[str]
    ):
        """Index when specific datasets exist within groups."""
        for group_name in group_names:
            group = h5_file[group_name]

            # Check if requested datasets exist in this group
            if all(key in group for key in self.ds_keys):
                # Get sample count and verify consistency
                sample_counts = [group[key].shape[0] for key in self.ds_keys]
                if len(set(sample_counts)) != 1:
                    raise ValueError(f"Sample count mismatch in group {group_name}")

                # Add all samples
                for sample_idx in range(sample_counts[0]):
                    self.sample_index.append(
                        (file_idx, group_name, "DATASET_PAIR", sample_idx)
                    )

    def _index_root_datasets(
        self,
        h5_file: h5py.File,
        file_idx: int,
        file_path: Path,
        dataset_names: List[str],
    ):
        """Index when datasets are at root level."""
        if self.ds_keys and all(key in dataset_names for key in self.ds_keys):
            # Dataset pairing at root level
            sample_counts = [h5_file[key].shape[0] for key in self.ds_keys]
            if len(set(sample_counts)) != 1:
                raise ValueError(f"Sample count mismatch in {file_path}")

            for sample_idx in range(sample_counts[0]):
                self.sample_index.append((file_idx, "ROOT", "DATASET_PAIR", sample_idx))
        else:
            # Individual datasets
            for dataset_name in dataset_names:
                if not self.ds_keys or dataset_name in self.ds_keys:
                    sample_count = h5_file[dataset_name].shape[0]
                    for sample_idx in range(sample_count):
                        self.sample_index.append(
                            (file_idx, "ROOT", dataset_name, sample_idx)
                        )

    def _load_tensor(
        self, file_idx: int, group_name: str, dataset_name: str
    ) -> torch.Tensor:
        """Load and cache tensor data."""
        cache_key = (file_idx, group_name, dataset_name)

        if cache_key not in self.data_cache:
            file_path = self.file_paths[file_idx]
            with h5py.File(file_path, "r") as h5_file:
                if group_name == "ROOT":
                    data = h5_file[dataset_name][:]
                elif group_name == "GROUP_PAIR":
                    # For group pairs, dataset_name is actually the dataset within groups
                    data = []
                    for group_key in self.ds_keys:
                        data.append(h5_file[group_key][dataset_name][:])
                else:
                    data = h5_file[group_name][dataset_name][:]

                if isinstance(data, list):
                    # Multiple tensors from group pairs
                    self.data_cache[cache_key] = [
                        ensure_tensor_format(d, self.is_3d) for d in data
                    ]
                else:
                    self.data_cache[cache_key] = ensure_tensor_format(data, self.is_3d)

        return self.data_cache[cache_key]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        file_idx, group_name, dataset_name, sample_idx = self.sample_index[index]

        if dataset_name == "DATASET_PAIR":
            # Multiple datasets form one sample
            file_path = self.file_paths[file_idx]
            with h5py.File(file_path, "r") as h5_file:
                samples = []
                for i, key in enumerate(self.ds_keys):
                    if group_name == "ROOT":
                        data = h5_file[key][sample_idx]
                    else:
                        data = h5_file[group_name][key][sample_idx]

                    tensor = ensure_tensor_format(data, self.is_3d)

                    # Apply transform if specified
                    if self.transform_names and i < len(self.transform_names):
                        transform = self.transformers.get(self.transform_names[i])
                        if transform:
                            tensor = transform(tensor)

                    samples.append(tensor)

                return tuple(samples)

        elif group_name == "GROUP_PAIR":
            # Group pairing
            tensors = self._load_tensor(file_idx, group_name, dataset_name)
            samples = []
            for i, tensor in enumerate(tensors):
                sample = tensor[sample_idx]
                if self.transform_names and i < len(self.transform_names):
                    transform = self.transformers.get(self.transform_names[i])
                    if transform:
                        sample = transform(sample)
                samples.append(sample)
            return tuple(samples)

        else:
            # Single dataset
            tensor = self._load_tensor(file_idx, group_name, dataset_name)
            sample = tensor[sample_idx]

            if self.transform_names and len(self.transform_names) == 1:
                transform = self.transformers.get(self.transform_names[0])
                if transform:
                    sample = transform(sample)

            return (sample,)

    def __len__(self) -> int:
        return len(self.sample_index)


def discover_data_files(args):
    """Discover data files based on configuration."""
    file_extension = getattr(args, "file_extension", "hdf")
    data_files = {dt: [] for dt in DatasetType}

    if args.base_path is not None:
        # Ensure no conflicting path specifications
        if any([args.train_files_path, args.test_files_path, args.val_files_path]):
            raise ValueError("Cannot specify both base_path and individual file paths!")

        base_path = Path(args.base_path)

        # Look for subdirectories first
        for dataset_type in [DatasetType.TRAIN, DatasetType.VAL, DatasetType.TEST]:
            subdir = base_path / dataset_type.value
            if subdir.is_dir():
                data_files[dataset_type] = sorted(subdir.glob(f"**/*.{file_extension}"))

        # If no subdirectories, use all files for training
        if not any(
            data_files[dt]
            for dt in [DatasetType.TRAIN, DatasetType.VAL, DatasetType.TEST]
        ):
            all_files = sorted(base_path.glob(f"**/*.{file_extension}"))
            if not all_files:
                raise ValueError(f"No {file_extension} files found in {base_path}")
            data_files[DatasetType.TRAIN] = all_files

    else:
        # Direct file specification
        if args.train_files_path:
            data_files[DatasetType.TRAIN] = [Path(p) for p in args.train_files_path]
        else:
            raise ValueError("No training data specified!")

        if args.val_files_path:
            data_files[DatasetType.VAL] = [Path(p) for p in args.val_files_path]

        if args.test_files_path:
            data_files[DatasetType.TEST] = [Path(p) for p in args.test_files_path]

    return data_files


def create_subset(dataset: data.Dataset, ratio: float, shuffle: bool) -> data.Dataset:
    """Create a subset of the dataset."""
    if ratio >= 1.0:
        return dataset

    subset_size = int(len(dataset) * ratio)
    if subset_size == 0:
        raise ValueError("Subset size cannot be zero!")

    if shuffle:
        subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
    else:
        subset = torch.utils.data.Subset(dataset, range(subset_size))

    return subset


def create_dataloader(
    dataset: data.Dataset, sampler: data.Sampler, args, batch_size_attr: str
):
    """Create a DataLoader if dataset is not empty."""
    if len(dataset) == 0:
        return None

    batch_size = getattr(args, batch_size_attr)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        sampler=sampler,
    )


def create_datasets(experiment, args=None):
    """
    Simplified data preparator for HDF5 files.

    Supports:
    - Files organized in subdirectories (train/val/test) or specified directly
    - Multiple HDF5 organizational patterns (groups, datasets, pairs)
    - Subset creation and train/val splitting
    - Distributed sampling
    """
    # Discover data files
    data_files = discover_data_files(args)

    # Determine what to load
    is_train = experiment.mode.op_mode.train
    is_test = experiment.mode.op_mode.test

    # Create datasets
    datasets = {}

    if is_train:
        iprint("Preparing training dataset")
        train_dataset = H5Dataset(
            file_paths=data_files[DatasetType.TRAIN],
            experiment=experiment,
            hdf_ds_def=args.hdf_ds_key_request,
            args=args,
        )

        if data_files[DatasetType.VAL]:
            iprint("Preparing validation dataset")
            val_dataset = H5Dataset(
                file_paths=data_files[DatasetType.VAL],
                experiment=experiment,
                hdf_ds_def=args.hdf_ds_key_request,
                args=args,
            )
        else:
            val_dataset = EmptyDataset()  # Empty dataset

        datasets["train"] = train_dataset
        datasets["val"] = val_dataset
    else:
        datasets["train"] = EmptyDataset()
        datasets["val"] = EmptyDataset()

    if is_test and data_files[DatasetType.TEST]:
        iprint("Preparing test dataset")
        test_ds_def = args.hdf_ds_key_request_test or args.hdf_ds_key_request
        datasets["test"] = H5Dataset(
            file_paths=data_files[DatasetType.TEST],
            experiment=experiment,
            hdf_ds_def=test_ds_def,
            args=args,
        )
    else:
        datasets["test"] = EmptyDataset()

    # Create subsets
    train_sub = create_subset(datasets["train"], args.subset_train, args.shuffle_train)
    val_sub = create_subset(datasets["val"], args.subset_val, args.shuffle_train)
    test_sub = create_subset(datasets["test"], args.subset_test, args.shuffle_test)

    # Handle train/val split if no validation set
    if len(val_sub) == 0 and args.val_from_train_ratio > 0 and len(train_sub) > 0:
        val_size = int(len(train_sub) * args.val_from_train_ratio)
        train_size = len(train_sub) - val_size

        if args.shuffle_train:
            train_sub, val_sub = random_split(train_sub, [train_size, val_size])
        else:
            train_all = train_sub
            train_sub = torch.utils.data.Subset(train_all, range(train_size))
            val_sub = torch.utils.data.Subset(
                train_all, range(train_size, train_size + val_size)
            )

    # Create samplers
    train_sampler = torch.utils.data.DistributedSampler(
        train_sub,
        num_replicas=experiment.world_size,
        rank=experiment.rank,
        shuffle=args.shuffle_train,
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_sub,
        num_replicas=experiment.world_size,
        rank=experiment.rank,
        shuffle=args.shuffle_val,
    )
    test_sampler = torch.utils.data.DistributedSampler(
        test_sub,
        num_replicas=experiment.world_size,
        rank=experiment.rank,
        shuffle=args.shuffle_test,
    )

    # Create data loaders
    train_loader = create_dataloader(train_sub, train_sampler, args, "train_batch_size")
    val_loader = create_dataloader(val_sub, val_sampler, args, "val_batch_size")
    test_loader = create_dataloader(test_sub, test_sampler, args, "test_batch_size")

    return {
        "train_data_loader": train_loader,
        "val_data_loader": val_loader,
        "test_data_loader": test_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "test_sampler": test_sampler,
        "n_train_samples": len(train_sub),
        "n_val_samples": len(val_sub),
        "n_test_samples": len(test_sub),
        "n_train_batches": len(train_loader) if train_loader else 0,
        "n_val_batches": len(val_loader) if val_loader else 0,
        "n_test_batches": len(test_loader) if test_loader else 0,
    }
