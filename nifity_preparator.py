import csv
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split

from fdq.ui_functions import iprint, wprint


SPLIT_KEYS = ("train", "val", "test")


def _load_hu_metadata(csv_path):
    """Map VolumeName -> (RescaleSlope, RescaleIntercept) from a CT-RATE-style metadata CSV."""
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["VolumeName"]] = (float(row["RescaleSlope"]), float(row["RescaleIntercept"]))
    return lookup


class NiftiDataset(data.Dataset):
    """Lazily loads one 3D volume per `.nii.gz` file.

    Nothing is read from disk until `__getitem__` is called - large NIfTI
    datasets are typically far too big to preload into RAM like the HDF5
    preparators do. If you want repeated epochs to be fast, enable fdq's
    dataset caching for this data source instead of relying on this loader
    to be fast on its own.

    CT-RATE's released .nii.gz files store raw scanner pixel values, not
    Hounsfield Units - their NIfTI headers have no usable scl_slope/scl_inter
    (both are NaN). The real per-scan RescaleSlope/RescaleIntercept only exist
    in the dataset's metadata CSVs, keyed by file name. Skipping this step
    silently produces garbage: e.g. train_73_a_1.nii.gz's raw values span
    [0, 30960] instead of the real (CSV-calibrated) HU range of roughly
    [-1024, a few thousand]. `hu_metadata`, if given, is applied as
    `hu = raw * slope + intercept` before anything else runs.
    """

    def __init__(self, file_paths, experiment, transform=None, hu_metadata=None):
        super().__init__()
        self.dataset_file_paths = file_paths
        self.transformers = experiment.transformers
        self.transform = transform
        self.hu_metadata = hu_metadata

    def __len__(self):
        return len(self.dataset_file_paths)

    def __getitem__(self, index):
        file_path = self.dataset_file_paths[index]
        img = nib.as_closest_canonical(nib.load(str(file_path)))
        volume = img.get_fdata().astype(np.float32)

        if self.hu_metadata is not None:
            calibration = self.hu_metadata.get(file_path.name)
            if calibration is None:
                raise ValueError(
                    f"No RescaleSlope/RescaleIntercept metadata found for '{file_path.name}'! "
                    "Check that 'metadata_dir' points at the matching CT-RATE dataset_stats directory."
                )
            slope, intercept = calibration
            volume = volume * slope + intercept

        # nibabel gives (X, Y, Z) in RAS+ orientation -> reorder to (D, H, W)
        volume = np.transpose(volume, (2, 1, 0))
        sample = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)

        if self.transform is None:
            return sample
        if isinstance(self.transform, (list, tuple)):
            # Multiple transforms of the *same* loaded volume, e.g. a clean view
            # and a degraded (blurred/etc.) view for conditional training - not
            # multiple different volumes.
            return tuple(t(sample) for t in self.transform)
        return self.transform(sample)


def _discover_files(split_dir, file_extension):
    if not os.path.isdir(split_dir):
        return []
    return sorted(Path(split_dir).glob(f"**/*{file_extension}"))


def _resolve_transform(experiment, transform_name):
    """Look up a transform_train/transform_test arg by name.

    A plain string resolves to a single transform (one tensor per sample, the
    existing behaviour). A list/tuple of names resolves to that many transforms
    applied to the *same* loaded volume, returned as a tuple - e.g. a clean and
    a degraded view of one volume for conditional training. Omit/null for no
    transform at all.
    """
    if not transform_name:
        return None
    if isinstance(transform_name, str):
        return experiment.transformers[transform_name]
    return [experiment.transformers[name] for name in transform_name]


def _select_subset(file_paths, max_samples, selection, seed):
    """Cap a file list to at most `max_samples` entries.

    `selection="sequential"` keeps the first `max_samples` files in sorted
    order (deterministic, no RNG involved). `selection="random"` draws a
    `seed`-based reproducible random subset (same seed -> same files every
    run); pass `seed: null` to get a different subset on every run instead.
    """
    if max_samples is None or max_samples >= len(file_paths):
        return file_paths

    if selection == "sequential":
        return file_paths[:max_samples]
    elif selection == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(file_paths, max_samples))
    else:
        raise ValueError(f"Unknown sample_selection '{selection}', must be 'sequential' or 'random'.")


def create_datasets(experiment, args=None):
    """NIfTI (.nii.gz) data preparator, e.g. for CT-RATE-style datasets laid out as
    <root_path>/<train_dir>/<patient>/<scan>/<scan>_<n>.nii.gz.

    Instead of listing individual files, point
    `root_path` at the dataset root and every matching file underneath the
    configured split directory is discovered and used directly.

    Preparator settings (under `args`):
    - root_path (str): dataset root directory.
    - train_dir (str): train-split subdirectory name under root_path (default "train").
    - val_dir (str, optional): val-split subdirectory name under root_path. If not set
      (the CT-RATE default), validation samples are carved out of the train files via
      `val_from_train_ratio`, exactly like the HDF5 preparators.
    - test_dir (str): test-split subdirectory name under root_path (default "valid",
      matching CT-RATE's held-out split name).
    - file_extension (str): file suffix to match (default ".nii.gz").
    - max_train_samples / max_val_samples / max_test_samples (int, optional): cap the
      number of files used per split. Omit/null = use all discovered files.
    - sample_selection (str): "sequential" (first N files in sorted order, deterministic)
      or "random" (seeded random subset). Default "sequential".
    - sample_seed (int, optional): seed for `sample_selection: random`. Default 42.
      Set to null for a different random subset on every run.
    - transform_train / transform_test (str or list[str], optional): name(s) of the
      transform(s) (from the experiment's `transforms:` block) applied to every loaded
      volume during training/val and testing respectively. A single name returns one
      transformed tensor per sample; a list of names applies each transform to the *same*
      loaded volume and returns a tuple of tensors (e.g. a clean + a degraded view of one
      volume, for conditional training - not multiple different volumes). Omit for raw
      (untransformed) volumes.
    - metadata_dir (str, optional but strongly recommended): directory containing the
      CT-RATE `<train_dir>_metadata.csv` / `<test_dir>_metadata.csv` files. Used to look
      up each file's real RescaleSlope/RescaleIntercept and convert raw stored values to
      Hounsfield Units - CT-RATE's .nii.gz headers do NOT carry usable rescale info
      (scl_slope/scl_inter are NaN), so without this every loaded volume is raw scanner
      output, not HU, and the HU_to_norm-style transforms silently operate on garbage.
    - subset_train / subset_val / subset_test (float): ratio of the (already capped)
      file list to use, same as in the other preparators.
    - val_from_train_ratio (float): fraction of train used for validation when no
      `val_dir` is configured.
    - pin_memory, num_workers, prefetch_factor, persistent_workers,
      multiprocessing_context, train_batch_size, val_batch_size, test_batch_size,
      shuffle_train, shuffle_val, shuffle_test, drop_last.
    """

    file_extension = args.get("file_extension", ".nii.gz")
    root_path = args.get("root_path")
    if root_path is None:
        raise ValueError("nifty_preparator requires 'root_path' to be set in the data args!")

    train_dir = args.get("train_dir", "train")
    val_dir = args.get("val_dir")
    test_dir = args.get("test_dir", "valid")

    sample_selection = args.get("sample_selection", "sequential")
    sample_seed = args.get("sample_seed", 42)

    num_workers = args.num_workers
    prefetch_factor = args.get("prefetch_factor", None)
    persistent_workers = args.get("persistent_workers")
    multiprocessing_context = args.get("multiprocessing_context", None)

    if multiprocessing_context not in {None, "fork", "spawn", "forkserver"}:
        raise ValueError(
            f"Invalid multiprocessing_context: {multiprocessing_context}. "
            "Must be one of None, 'fork', 'spawn', or 'forkserver'."
        )
    if multiprocessing_context is None and experiment.is_distributed() and num_workers > 0:
        multiprocessing_context = "forkserver"

    if num_workers > 0 and persistent_workers is None:
        persistent_workers = True
        wprint(
            f"DataLoader num_workers={num_workers}: setting persistent_workers=True. "
            "Set persistent_workers: false in the data args to disable this."
        )
    elif num_workers == 0:
        persistent_workers = False

    data_files = {
        "train": _discover_files(os.path.join(root_path, train_dir), file_extension),
        "val": _discover_files(os.path.join(root_path, val_dir), file_extension) if val_dir else [],
        "test": _discover_files(os.path.join(root_path, test_dir), file_extension),
    }

    for split_name in SPLIT_KEYS:
        max_samples = args.get(f"max_{split_name}_samples")
        data_files[split_name] = _select_subset(data_files[split_name], max_samples, sample_selection, sample_seed)

    if len(data_files["train"]) == 0:
        raise ValueError(f"No '{file_extension}' files found in {os.path.join(root_path, train_dir)} !")

    metadata_dir = args.get("metadata_dir")
    hu_metadata = None
    if metadata_dir is not None:
        hu_metadata = {}
        for split_dir_name in {train_dir, test_dir, val_dir} - {None}:
            csv_path = os.path.join(metadata_dir, f"{split_dir_name}_metadata.csv")
            if os.path.isfile(csv_path):
                hu_metadata.update(_load_hu_metadata(csv_path))
        if not hu_metadata:
            raise ValueError(f"No '*_metadata.csv' files found for any split under metadata_dir={metadata_dir} !")
    else:
        wprint(
            "No 'metadata_dir' configured for the nifty_preparator dataset: loaded volumes will be raw "
            "stored values, NOT calibrated Hounsfield Units! Set 'metadata_dir' unless you know your "
            "NIfTI files already carry correct scl_slope/scl_inter calibration."
        )

    is_train = experiment.mode.op_mode.train
    if is_train:
        iprint("Preparing training dataset")

        transform = _resolve_transform(experiment, args.get("transform_train"))

        train = NiftiDataset(
            file_paths=data_files["train"], experiment=experiment, transform=transform, hu_metadata=hu_metadata
        )

        if len(data_files["val"]) > 0:
            iprint("Preparing validation dataset")
            val = NiftiDataset(
                file_paths=data_files["val"], experiment=experiment, transform=transform, hu_metadata=hu_metadata
            )
        else:
            val = []
    else:
        iprint("This is a test run - no training and validation sets are being generated.")
        train = []
        val = []

        transform = _resolve_transform(experiment, args.get("transform_test"))

    is_test = experiment.mode.op_mode.test
    if is_test and len(data_files["test"]) > 0:
        iprint("Preparing test dataset")
        test = NiftiDataset(
            file_paths=data_files["test"], experiment=experiment, transform=transform, hu_metadata=hu_metadata
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
        wprint(f"\n\nWARNING: Training subset is only {100 * (n_train_sub / n_train):.1f}% original training set!\n\n")

    n_val_sub = int(n_val * val_subratio)
    n_test_sub = int(n_test * test_subratio)

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

    # split train into val and train (if no dedicated val set is defined)
    train_ratio = args.val_from_train_ratio
    if len(val_sub) == 0 and train_ratio != 0:
        n_val_sub = int(n_train_sub * train_ratio)
        n_train_sub = int(n_train_sub - n_val_sub)

        if not args.shuffle_train:
            train_sub_all = train_sub
            train_sub = torch.utils.data.Subset(train_sub_all, range(n_train_sub))
            val_sub = torch.utils.data.Subset(train_sub_all, range(n_train_sub, n_val_sub + n_train_sub))
        else:
            train_sub, val_sub = random_split(train_sub, [n_train_sub, n_val_sub])
    elif is_train:
        wprint("WARNING: no validation set defined!")

    train_sampler = torch.utils.data.DistributedSampler(
        train_sub, num_replicas=experiment.world_size, rank=experiment.rank, shuffle=args.shuffle_train
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_sub, num_replicas=experiment.world_size, rank=experiment.rank, shuffle=args.shuffle_val
    )
    test_sampler = torch.utils.data.DistributedSampler(
        test_sub, num_replicas=experiment.world_size, rank=experiment.rank, shuffle=args.shuffle_test
    )

    if len(train_sub) > 0:
        train_data_loader = torch.utils.data.DataLoader(
            train_sub,
            batch_size=args.train_batch_size,
            pin_memory=args.pin_memory,
            num_workers=num_workers,
            drop_last=args.drop_last,
            sampler=train_sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        train_data_loader = None

    if len(val_sub) > 0:
        val_data_loader = torch.utils.data.DataLoader(
            val_sub,
            batch_size=args.val_batch_size,
            pin_memory=args.pin_memory,
            num_workers=num_workers,
            drop_last=args.drop_last,
            sampler=val_sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        val_data_loader = None

    if len(test_sub) > 0:
        test_data_loader = torch.utils.data.DataLoader(
            test_sub,
            batch_size=args.test_batch_size,
            pin_memory=args.pin_memory,
            num_workers=num_workers,
            drop_last=args.drop_last,
            sampler=test_sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
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
        "n_train_batches": len(train_data_loader) if train_data_loader is not None else 0,
        "n_val_batches": len(val_data_loader) if val_data_loader is not None else 0,
        "n_test_batches": len(test_data_loader) if test_data_loader is not None else 0,
    }
