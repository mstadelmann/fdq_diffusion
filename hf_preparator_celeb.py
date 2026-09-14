import os

import torch
from datasets import load_dataset
from torch.utils import data
from torchvision.transforms.functional import pil_to_tensor

from fdq.ui_functions import iprint, wprint

HF_DATASET_NAME = "bitmind/celeb-a-hq"
HF_SOURCE_SPLIT = "train"  # the only split this dataset ships


class HFCelebDataset(data.Dataset):
    """Wraps (a subset of) the Hugging Face bitmind/celeb-a-hq dataset.

    Only the 'image' column is used: this loader is for unconditional face
    generation, no labels are needed.
    """

    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = pil_to_tensor(self.hf_dataset[index]["image"].convert("RGB"))
        if self.transform is None:
            return sample
        if isinstance(self.transform, (list, tuple)):
            return tuple(t(sample) for t in self.transform)
        return self.transform(sample)

    def __len__(self):
        return len(self.hf_dataset)


def _resolve_transform(experiment, transform_name):
    if not transform_name:
        return None
    if isinstance(transform_name, str):
        return experiment.transformers[transform_name]
    return [experiment.transformers[name] for name in transform_name]


def _make_subset(dataset, subratio, shuffle):
    n = len(dataset)
    n_sub = int(n * subratio)
    if n_sub == 0:
        raise ValueError("Current settings lead to an empty dataset subset!")
    if n_sub == n:
        return dataset
    if not shuffle:
        return torch.utils.data.Subset(dataset, range(n_sub))
    sub, _ = torch.utils.data.random_split(dataset, [n_sub, n - n_sub])
    return sub


def create_datasets(experiment, args=None):
    """Downloads (once) and prepares the Hugging Face bitmind/celeb-a-hq dataset.

    https://huggingface.co/datasets/bitmind/celeb-a-hq

    `hf_cache_dir` is used as the `datasets` library cache_dir: the first run
    downloads and converts the parquet shards there; every subsequent run
    (any experiment, sharing that same path) reuses the on-disk cache with
    no re-download.

    Deliberately NOT named 'base_path': fdq's generic
    fdqExperiment.copy_data_to_scratch() rsyncs any data arg literally named
    base_path into the job's ephemeral scratch dir and rewrites the arg to
    point there instead - which would silently re-download this dataset into
    scratch on every single job and lose it again when the job ends. This
    function never reads args.base_path; the corresponding config still
    declares `base_path: null` as a no-op placeholder, since that same
    copy_data_to_scratch() reads it via `.` (not `.get()`) unconditionally on
    every data source and raises ConfigAttributeError if the key is missing
    entirely - `null` satisfies that check (None -> copy skipped) without
    this preparator ever using the value.

    This dataset ships a single 30k-image 'train' split with no built-in
    val/test - `val_from_train_ratio` and `test_from_train_ratio` carve fixed,
    non-overlapping val/test slices off the *end* of that split (by row index,
    not randomly), so the same held-out images are used whether this function
    is called from the training job or from a later, separate test job.
    """
    cache_dir = os.path.expanduser(args.hf_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    num_workers = args.num_workers
    prefetch_factor = args.get("prefetch_factor", None)
    persistent_workers = args.get("persistent_workers")
    if num_workers > 0 and persistent_workers is None:
        persistent_workers = True
        wprint(
            f"DataLoader num_workers={num_workers}: setting persistent_workers=True. "
            "Set persistent_workers: false in the data args to disable this."
        )
    elif num_workers == 0:
        persistent_workers = False

    is_train = experiment.mode.op_mode.train
    is_test = experiment.mode.op_mode.test

    iprint(f"Loading '{HF_DATASET_NAME}' from Hugging Face (cache: {cache_dir})...")
    full = load_dataset(HF_DATASET_NAME, split=HF_SOURCE_SPLIT, cache_dir=cache_dir)

    n = len(full)
    n_test = int(n * args.test_from_train_ratio)
    n_val = int(n * args.val_from_train_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"val_from_train_ratio ({args.val_from_train_ratio}) + test_from_train_ratio "
            f"({args.test_from_train_ratio}) leave no training images out of {n} total!"
        )

    train_sub = []
    val_sub = []
    test_sub = []

    if is_train:
        transform = _resolve_transform(experiment, args.transform)

        train_hf = full.select(range(n_train))
        val_hf = full.select(range(n_train, n_train + n_val))

        train = HFCelebDataset(train_hf, transform=transform)
        val = HFCelebDataset(val_hf, transform=transform)

        train_sub = _make_subset(train, args.subset_train, args.shuffle_train)
        val_sub = _make_subset(val, args.subset_val, args.shuffle_val)
    else:
        iprint("This is a test run - no training and validation sets are being generated.")

    if is_test:
        test_transform = _resolve_transform(experiment, args.transform_test)
        test_hf = full.select(range(n_train + n_val, n))
        test = HFCelebDataset(test_hf, transform=test_transform)
        test_sub = _make_subset(test, args.subset_test, args.shuffle_test)

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
            num_workers=num_workers,
            drop_last=args.drop_last,
            sampler=train_sampler,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
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
        "n_val_samples": len(val_sub),
        "n_test_samples": len(test_sub),
        "n_train_batches": len(train_data_loader) if train_data_loader is not None else 0,
        "n_val_batches": len(val_data_loader) if val_data_loader is not None else 0,
        "n_test_batches": len(test_data_loader) if test_data_loader is not None else 0,
    }
