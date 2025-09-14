import os
from enum import Enum
from pathlib import Path
import bisect
import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split
import itertools

from fdq.ui_functions import startProgBar, iprint, wprint


class DatasetType(Enum):
    Train = "train"
    Val = "val"
    Test = "test"
    All = "all"


class H5Dataset(data.Dataset):
    """
    HDF5 dataset loader.

    Data itself is stored in self.hdf_data_tensor.
    self.hdf_data_tensor is a list of lists of lists of tensors.
    where the outermost list is per file, the second list is per group, and the innermost list is per dataset.
    The tensors are the actual data samples in the files (e.g. [C,H,W] or [D,C,H,W] format).
    """

    def __init__(self, file_paths, experiment, transform=None):
        """
        file_paths (list): List of file paths.
        data_transform : already initialized data transformer.
        data_is_3d (bool, optional): Defaults to False.
        """

        super().__init__()

        self.dataset_file_paths = file_paths
        self.transformers = experiment.transformers
        self.dataset_nb_samples = None
        self.nb_samples_per_file = []
        self.transform = transform
        nb_source_files = len(self.dataset_file_paths)

        pbar = startProgBar(
            nb_source_files,
            f"Processing files (Total nb files: {nb_source_files})",
            is_active=nb_source_files > 1,
        )

        for file_idx, fp in enumerate(self.dataset_file_paths):
            pbar.update(file_idx)
            with h5py.File(fp) as h5_file:
                root_keys = list(h5_file.keys())
                self.nb_samples_per_file.append(len(root_keys))

        pbar.finish()

        self.nb_samples_per_file_acc = [0]
        self.nb_samples_per_file_acc.extend(
            list(itertools.accumulate(self.nb_samples_per_file))
        )

        self.total_nb_samples = self.nb_samples_per_file_acc[-1]

    def __getitem__(self, index):
        file_idx = bisect.bisect_right(self.nb_samples_per_file_acc, index) - 1
        local_index = index - self.nb_samples_per_file_acc[file_idx]

        with h5py.File(self.dataset_file_paths[file_idx]) as h5_file:
            root_keys = list(h5_file.keys())
            sample = torch.tensor(np.array(h5_file[root_keys[local_index]]))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.total_nb_samples


def create_datasets(experiment, args=None):
    """
    This is a very basic data preparator for HDF files, e.g. for the celebrity dataset,
    where each file contains one or many images, without any labels.
    This dataloader is very slow and should be used with an additional caching mechanism!
    """

    file_extension = args.get("file_extension", "hdf")
    data_files = {}
    data_basePath = args.base_path

    if args.base_path is not None:
        if (
            args.train_files_path is not None
            or args.test_files_path is not None
            or args.val_files_path is not None
        ):
            raise ValueError(
                "This combination of data path definitions is not allowed!!"
            )

        for data_dir in DatasetType:
            if data_dir == DatasetType.All:
                data_path = Path(data_basePath)
            else:
                data_path = Path(os.path.join(data_basePath, data_dir.value))

            if os.path.isdir(data_path):
                data_files[data_dir] = sorted(data_path.glob(f"**/*.{file_extension}"))
            else:
                data_files[data_dir] = []

        if len(data_files[DatasetType.All]) == 0:
            iprint(f"File extension filter: {file_extension}")
            raise ValueError(f"No data files found in {args.base_path} !")

        if len(data_files[DatasetType.Train]) == 0:
            if (
                len(data_files[DatasetType.Test]) != 0
                or len(data_files[DatasetType.Val]) != 0
            ):
                raise ValueError(
                    "No train set defined, but test or validation set! This case is currently not supported!"
                )
            data_files[DatasetType.Train] = data_files[DatasetType.All]

    elif data_basePath is None:
        if args.train_files_path is not None:
            data_files[DatasetType.Train] = [Path(p) for p in args.train_files_path]
        else:
            raise ValueError("No Train Data Files defined!")

        if args.val_files_path is not None:
            data_files[DatasetType.Val] = [Path(p) for p in args.val_files_path]
        else:
            data_files[DatasetType.Val] = []

        if args.test_files_path is not None:
            data_files[DatasetType.Test] = [Path(p) for p in args.test_files_path]
        else:
            data_files[DatasetType.Test] = []

    is_train = experiment.mode.op_mode.train
    if is_train:
        iprint("Preparing training dataset")

        if len(args.hdf_ds_key_request) == 1 and "*" in args.hdf_ds_key_request[0]:
            transform_name = args.hdf_ds_key_request[0]["*"]
            transform = experiment.transformers[transform_name]
        else:
            transform = None

        train = H5Dataset(
            file_paths=data_files[DatasetType.Train],
            experiment=experiment,
            transform=transform,
        )

        if len(data_files[DatasetType.Val]) > 0:
            iprint("Preparing validation dataset")
            val = H5Dataset(
                file_paths=data_files[DatasetType.Val],
                experiment=experiment,
                transform=transform,
            )
        else:
            val = []
    else:
        iprint(
            "This is a test run - no training and validation sets are being generated."
        )
        train = []
        val = []

        if (
            len(args.hdf_ds_key_request_test) == 1
            and "*" in args.hdf_ds_key_request_test[0]
        ):
            transform_name = args.hdf_ds_key_request_test[0]["*"]
            transform = experiment.transformers[transform_name]
        else:
            transform = None

    is_test = experiment.mode.op_mode.test
    if is_test and len(data_files[DatasetType.Test]) > 0:
        iprint("Preparing test dataset")
        test = H5Dataset(
            file_paths=data_files[DatasetType.Test],
            experiment=experiment,
            transform=transform,
        )
    else:
        test = []

    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    train_subratio = args.subset_train
    val_subratio = args.subset_val
    test_subratio = args.subset_test

    n_train_sub = int(n_train * train_subratio)
    if n_train_sub == 0 and is_train:
        raise ValueError("Current settings lead to empty train subset!")
    if n_train_sub < 0.5 * n_train and is_train:
        wprint(
            f"\n\nWARNING: Training subset is only {100 * (n_train_sub / n_train):.1f}% original training set!\n\n"
        )

    n_val_sub = int(n_val * val_subratio)
    n_test_sub = int(n_test * test_subratio)

    # compute subset per category
    if n_train == n_train_sub:
        train_sub = train
    else:
        if not args.shuffle_train:
            train_sub = torch.utils.data.Subset(train, range(n_train_sub))
        else:
            train_sub, _ = random_split(train, [n_train_sub, n_train - n_train_sub])

    if n_val == n_val_sub:
        val_sub = val
    else:
        if not args.shuffle_train:
            val_sub = torch.utils.data.Subset(val, range(n_val_sub))
        else:
            val_sub, _ = random_split(val, [n_val_sub, n_val - n_val_sub])

    if n_test == n_test_sub:
        test_sub = test
    else:
        if not args.shuffle_test:
            test_sub = torch.utils.data.Subset(test, range(n_test_sub))
        else:
            test_sub, _ = random_split(test, [n_test_sub, n_test - n_test_sub])

    # spilt train into val and train (if no val set is defined)
    train_ratio = args.val_from_train_ratio
    if len(val_sub) == 0 and train_ratio != 0:
        n_val_sub = int(n_train_sub * train_ratio)
        n_train_sub = int(n_train_sub - n_val_sub)

        if not args.shuffle_train:
            # gen = torch.Generator().manual_seed(42)
            # train_sub, val_sub = random_split(train_sub, [n_train_sub, n_val_sub], generator=gen)

            # additional copy uses more RAM, but keeps the first n samples in train set (for testing...)
            train_sub_all = train_sub
            train_sub = torch.utils.data.Subset(train_sub_all, range(n_train_sub))
            val_sub = torch.utils.data.Subset(
                train_sub_all, range(n_train_sub, n_val_sub + n_train_sub)
            )
        else:
            train_sub, val_sub = random_split(train_sub, [n_train_sub, n_val_sub])

    elif is_train:
        wprint("WARNING: no validation set defined!")

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

    if len(train_sub) > 0:
        train_data_loader = torch.utils.data.DataLoader(
            train_sub,
            batch_size=args.train_batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sampler=train_sampler,
        )
    else:
        train_data_loader = None

    if len(val_sub) > 0:
        val_data_loader = torch.utils.data.DataLoader(
            val_sub,
            batch_size=args.val_batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sampler=val_sampler,
        )
    else:
        val_data_loader = None

    if len(test_sub) > 0:
        test_data_loader = torch.utils.data.DataLoader(
            test_sub,
            batch_size=args.test_batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sampler=test_sampler,
        )
    else:
        test_data_loader = None

    return {
        "train_data_loader": train_data_loader,
        "val_data_loader": val_data_loader,
        "test_data_loader": test_data_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "test_sampler": test_sampler,
        "n_train_samples": len(train_sub),
        "n_val_samples": len(val_sub) if val_data_loader is not None else 0,
        "n_test_samples": len(test_sub),
        "n_train_batches": (
            len(train_data_loader) if train_data_loader is not None else 0
        ),
        "n_val_batches": len(val_data_loader) if val_data_loader is not None else 0,
        "n_test_batches": len(test_data_loader) if test_data_loader is not None else 0,
    }
