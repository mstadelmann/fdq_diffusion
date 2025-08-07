import os
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import random_split


from fdq.ui_functions import startProgBar, iprint, wprint


class DatasetType(Enum):
    Train = "train"
    Val = "val"
    Test = "test"
    All = "all"


def to_BCHW(tensor):
    """
    Reshape (unsqueeze) tensor to BCHW format.
    -> Add dimensions if batch or channel is missing.
    """
    if not isinstance(tensor, torch.Tensor):
        # to prevent slow conversion speed warning
        tensor = torch.tensor(np.array(tensor))
    if len(tensor.shape) == 3:
        if tensor.shape[0] > 3:
            # assume BHW
            tensor = tensor.unsqueeze(1)
        else:
            # assume CHW
            tensor = tensor.unsqueeze(0)
    elif len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def to_BCDHW(tensor):
    """
    Reshape (unsqueeze) tensor to BCDHW format.
    -> Add dimensions if batch or channel is missing.
    WIP for certain 3D data formats.
    """
    if not isinstance(tensor, torch.Tensor):
        # to prevent slow conversion speed warning
        tensor = torch.tensor(np.array(tensor))
    if len(tensor.shape) == 3:
        # assume [D,H,W]
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise NotImplementedError(
            "Update HDF loader to handle this input data format! E3"
        )
    return tensor


def _cleanup_hdf_keys(requests):
    if requests is None:
        return None
    if not isinstance(requests, list):
        raise ValueError("hdf_ds_key_request must be [str,..] or [{str: str},..] !")
    if isinstance(requests[0], dict):
        if len(requests[0]) != 1:
            raise ValueError(
                "hdf_ds_key_request must be [{str: str},..] with one key per dict!"
            )
        keys, values = zip(*[(k, v) for d in requests for k, v in d.items()])
        return list(keys), list(values)
    else:
        raise NotImplementedError


class H5Dataset(data.Dataset):
    """
    HDF5 dataset loader.

    Data itself is stored in self.hdf_data_tensor.
    self.hdf_data_tensor is a list of lists of lists of tensors.
    where the outermost list is per file, the second list is per group, and the innermost list is per dataset.
    The tensors are the actual data samples in the files (e.g. [C,H,W] or [D,C,H,W] format).
    """

    def __init__(self, file_paths, experiment, hdf_ds_def=None, args=None):
        """
        file_paths (list): List of file paths.
        data_transform : already initialized data transformer.
        hdf_ds_key_request (list): List of keys to load from HDF file.
        data_is_3d (bool, optional): Defaults to False.
        """

        super().__init__()

        data_is_3d = args.data_is_3d
        self.hdf_ds_key_request, self.transform_names = _cleanup_hdf_keys(hdf_ds_def)
        self.dataset_file_paths = file_paths
        self.dataset_filenames = []
        self.transformers = experiment.transformers

        self.av_group_names = None
        self.nb_smpls_per_ds_per_grp_per_fn = []

        self.dataset_nb_samples = []
        self.dataset_shape = []
        self.dataset_data = []

        self.groups_are_sample_pairs = None
        # groups_are_sample_pairs = True
        # means that one "data sample" corresponds
        # to one dataset in each group
        self.datasets_are_sample_pairs = None
        # datasets_are_sample_pairs = True means that one "data sample" corresponds
        # to one index in each HDF dataset
        # e.g. one dataset for the image, one dataset for the mask

        self.hdf_data_tensor = []  # -> [[[]]]  -> list per file, per group, per dataset

        nb_source_files = len(self.dataset_file_paths)
        pbar_over_files = nb_source_files > 1
        pbar = startProgBar(
            nb_source_files,
            f"Processing files (Total nb files: {nb_source_files})",
            is_active=pbar_over_files,
        )

        for file_idx, fp in enumerate(self.dataset_file_paths):
            pbar.update(file_idx)

            self.dataset_filenames.append(fp.stem)
            self.current_file_idx = file_idx
            self.current_fp = fp

            h5_file = h5py.File(fp)
            self.current_h5file = h5_file
            root_keys = list(h5_file.keys())
            has_key_requests = (
                isinstance(self.hdf_ds_key_request, list)
                and len(self.hdf_ds_key_request) > 0
            )

            datasets_are_sample_pairs = has_key_requests and all(
                elt in root_keys for elt in self.hdf_ds_key_request
            )
            # datasets_are_sample_pairs: one file has two datasets. One for images, one for labels.
            # image.shape = [10,1,30,30] and label.shape = [10,1,1,1]
            # see test scenario 1
            # pairs can also triplets, quadruplets or more, e.g. image, label, mask
            if self.datasets_are_sample_pairs is None:
                self.datasets_are_sample_pairs = datasets_are_sample_pairs
            elif self.datasets_are_sample_pairs != datasets_are_sample_pairs:
                raise ValueError(
                    "Datasets are either sample pairs or not! This cannot change!"
                )

            has_groups = isinstance(h5_file[root_keys[0]], h5py._hl.group.Group)

            if has_groups:
                group_names = root_keys
            else:
                group_names = ["ROOTGROUP"]

            groups_are_sample_pairs = has_key_requests and all(
                elt in group_names for elt in self.hdf_ds_key_request
            )
            # groups_are_sample_pairs: file has two groups, each with multiple datasets.
            # a sample is a pair of datasets, one from each group.
            # see test scenario 4
            # pairs can also triplets, quadruplets or more, e.g. image, label, mask
            if self.groups_are_sample_pairs is None:
                self.groups_are_sample_pairs = groups_are_sample_pairs
            elif self.groups_are_sample_pairs != groups_are_sample_pairs:
                raise ValueError(
                    "Groups are either sample pairs or not! This cannot change!"
                )

            if self.av_group_names is None:
                self.av_group_names = group_names

            if self.groups_are_sample_pairs:
                nb_samples_per_dataset_per_group = self._process_group_samples(
                    h5_file=h5_file,
                    fp=fp,
                    print_pbar=not pbar_over_files,
                    data_is_3d=data_is_3d,
                )

            elif self.datasets_are_sample_pairs:
                if data_is_3d:
                    raise NotImplementedError(
                        "Update HDF loader to handle this input data format! E1"
                    )
                nb_samples_per_dataset_per_group = [
                    [
                        self._process_dataset_samples(
                            h5_file=h5_file, fp=fp, print_pbar=not pbar_over_files
                        )
                    ]
                ]

            else:
                if data_is_3d:
                    raise NotImplementedError(
                        "Update HDF loader to handle this input data format! E2"
                    )
                nb_samples_per_dataset_per_group = self._process_samples(
                    group_names, root_keys, h5_file, fp, print_pbar=not pbar_over_files
                )

            self.nb_smpls_per_ds_per_grp_per_fn.append(nb_samples_per_dataset_per_group)

        pbar.finish()

        # pad nb_smpls_per_ds_per_grp_per_fn to have same number of groups per file
        # --> this is necessary to convert the list of lists of lists to a tensor !
        max_nb_groups = max(
            len(gn) for fn in self.nb_smpls_per_ds_per_grp_per_fn for gn in fn
        )
        for fni, _ in enumerate(self.nb_smpls_per_ds_per_grp_per_fn):
            for gni in range(len(self.nb_smpls_per_ds_per_grp_per_fn[fni])):
                if len(self.nb_smpls_per_ds_per_grp_per_fn[fni][gni]) < max_nb_groups:
                    nb_padding = max_nb_groups - len(
                        self.nb_smpls_per_ds_per_grp_per_fn[fni][gni]
                    )
                    self.nb_smpls_per_ds_per_grp_per_fn[fni][gni].extend(
                        nb_padding * [0]
                    )

        self.nb_smpls_per_ds_per_grp_per_fn = torch.tensor(
            self.nb_smpls_per_ds_per_grp_per_fn
        )
        self.total_nb_samples = int(torch.sum(self.nb_smpls_per_ds_per_grp_per_fn))

    def _process_group_samples(self, h5_file, fp, print_pbar=False, data_is_3d=False):
        nb_samples_per_dataset_per_group = []
        tensor_per_group = []

        for req_group in self.hdf_ds_key_request:
            group_datasets = h5_file[req_group].keys()
            tensor_per_dataset = []

            pbar = startProgBar(
                len(group_datasets),
                f"Proc. {os.path.basename(fp)}, Gr. Name: {req_group} (tot. nb. ds in gr.: {len(group_datasets)})",
                is_active=print_pbar,
            )

            for i, gds in enumerate(group_datasets):
                pbar.update(i)
                if data_is_3d:
                    tensor_per_dataset.append(to_BCDHW(h5_file[req_group][gds]))
                else:
                    tensor_per_dataset.append(to_BCHW(h5_file[req_group][gds]))

            tensor_per_group.append(tensor_per_dataset)
            nb_samples_per_dataset_per_group.append(
                [t.shape[0] for t in tensor_per_dataset]
            )
            pbar.finish()

        self.hdf_data_tensor.append(tensor_per_group)

        group_sample_count_matches = all(
            c == nb_samples_per_dataset_per_group[0]
            for c in nb_samples_per_dataset_per_group
        )

        if not group_sample_count_matches:
            raise ValueError(f"Number of samples per group in {fp} does not match!")

        return [nb_samples_per_dataset_per_group[0]]

    def _process_dataset_samples(self, h5_file, fp, print_pbar=False):
        pbar = startProgBar(
            len(self.hdf_ds_key_request),
            f"Processing {os.path.basename(fp)} (tot. nb. ds: {len(self.hdf_ds_key_request)})",
            is_active=print_pbar,
        )

        tensor_per_dataset = []
        # load all required hdf files
        for i, req_ds in enumerate(self.hdf_ds_key_request):
            pbar.update(i)
            tensor_per_dataset.append(to_BCHW(h5_file[req_ds]))

        tensor_per_group = [tensor_per_dataset]
        self.hdf_data_tensor.append(tensor_per_group)

        if not len({t.shape[0] for t in tensor_per_dataset}) == 1:
            raise ValueError(f"Number of samples per dataset in {fp} does not match!")

        pbar.finish()
        return tensor_per_group[0][0].shape[0]

    def _process_samples(self, group_names, root_keys, h5_file, fp, print_pbar=False):
        nb_samples_per_dataset_per_group = []
        tensor_per_group = []
        for hdf_group in group_names:
            nb_samples_per_dataset = []

            if hdf_group == "ROOTGROUP":
                pbar = startProgBar(
                    len(root_keys),
                    f"Processing {os.path.basename(fp)}",
                    is_active=print_pbar,
                )

                tensor_per_ds = []
                for i, ds in enumerate(root_keys):
                    pbar.update(i)

                    ds = to_BCHW(h5_file[ds])
                    nb_samples_per_dataset.append(ds.shape[0])
                    tensor_per_ds.append(ds)
                tensor_per_group.append(tensor_per_ds)
            else:
                group_keys = list(h5_file[hdf_group].keys())
                if self.hdf_ds_key_request is None:
                    # assume each ds to be a sample
                    raise ValueError("This case is currently not supported!")
                    # for ds in group_keys:
                    #     if isinstance(h5_file[hdf_group][ds], h5py._hl.group.Group):
                    #         raise ValueError(f"Groups in {fp} are nested! This is currently not supported!")
                    #     ds_shape = h5_file[hdf_group][ds].shape
                    #     nb_samples_per_dataset.append(ds_shape[0])

                # else:
                # look for specific datasets
                req_ds_present = all(
                    req_ds in group_keys for req_ds in self.hdf_ds_key_request
                )
                if not req_ds_present:
                    raise ValueError(
                        f"Requested ds_keys {self.hdf_ds_key_request} not present in {fp}!"
                    )

                samples_per_req_ds = []
                tensor_per_ds = []

                pbar = startProgBar(
                    len(self.hdf_ds_key_request),
                    f"Processing {os.path.basename(fp)}",
                    is_active=print_pbar,
                )

                for i, req_ds in enumerate(self.hdf_ds_key_request):
                    pbar.update(i)
                    ds = to_BCHW(h5_file[hdf_group][req_ds])
                    samples_per_req_ds.append(ds.shape[0])
                    tensor_per_ds.append(ds)

                tensor_per_group.append(tensor_per_ds)

                if not all(x == samples_per_req_ds[0] for x in samples_per_req_ds):
                    raise ValueError(
                        f"Number of samples per key in {fp} does not match!"
                    )

                nb_samples_per_dataset.append(samples_per_req_ds[0])
            pbar.finish()

            nb_samples_per_dataset_per_group.append(nb_samples_per_dataset)
        self.hdf_data_tensor.append(tensor_per_group)

        return nb_samples_per_dataset_per_group

    def _get_sample(self, index, file_idx):
        # Select group
        nb_groups_in_file = self.nb_smpls_per_ds_per_grp_per_fn[file_idx].shape[0]

        for group_idx in range(nb_groups_in_file):
            nb_samples_prev_files = int(
                torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[:file_idx])
            )
            nb_samples_current_file = int(
                torch.sum(
                    self.nb_smpls_per_ds_per_grp_per_fn[file_idx][: group_idx + 1]
                )
            )
            nb_samples = nb_samples_prev_files + nb_samples_current_file

            if index < nb_samples:
                break

        # Select dataset (image-label pair counts as 1 dataset!)
        nb_datasets_in_group = len(
            self.nb_smpls_per_ds_per_grp_per_fn[file_idx, group_idx, :]
        )

        for dataset_idx in range(nb_datasets_in_group):
            nb_samples_prev_files = int(
                torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[:file_idx])
            )
            nb_samples_prev_groups = int(
                torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[file_idx][:group_idx])
            )
            nb_samples_current_ds = int(
                torch.sum(
                    self.nb_smpls_per_ds_per_grp_per_fn[file_idx][group_idx][
                        : dataset_idx + 1
                    ]
                )
            )
            nb_samples = (
                nb_samples_prev_files + nb_samples_prev_groups + nb_samples_current_ds
            )

            if index < nb_samples:
                break

        # Select sample
        nb_samples_in_dataset = int(
            self.nb_smpls_per_ds_per_grp_per_fn[file_idx, group_idx, dataset_idx]
        )

        if nb_samples_in_dataset == 1:
            sample_idx = 0
        else:
            prev_files_offset = int(
                torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[:file_idx])
            )
            if group_idx == 0:
                group_offset = 0
            else:
                raise ValueError("Expecting no groups in this configuration!!")

            if dataset_idx == 0:
                dataset_offset = 0
            else:
                raise ValueError("Expecting only one dataset in this configuration!!")

            sample_idx = index - prev_files_offset - group_offset - dataset_offset

        if self.av_group_names == ["ROOTGROUP"]:
            ds = self.hdf_data_tensor[file_idx][0][dataset_idx]

            if self.transform_names is None or self.hdf_ds_key_request is None:
                return tuple(ds[sample_idx, :].unsqueeze(0))

            elif len(self.transform_names) != 1 or len(self.hdf_ds_key_request) != 1:
                raise NotImplementedError("This case is currently not supported!")

            elif self.hdf_ds_key_request[0] == "*":
                return tuple(
                    self.transformers[self.transform_names[0]](
                        ds[sample_idx, :].unsqueeze(0)
                    )
                )

        elif self.hdf_ds_key_request is not None:
            raise NotImplementedError("This case is currently not supported! E1")
            # all_data = []
            # for key_idx, req_key in enumerate(self.hdf_ds_key_request):
            #     # apply transformation per hdf_ds_key_request
            #     transform = self._get_transform(req_key)
            #     ds = self.hdf_data_tensor[file_idx][group_idx][key_idx]
            #     ds = ds.squeeze(0)  # TODO: why is is CHW only?
            #     if transform is None:
            #         all_data.append(ds)
            #     else:
            #         all_data.append(transform(ds))

            # return tuple(all_data)

        else:
            raise ValueError(
                f"ERROR, {self.dataset_file_paths[file_idx]} structure does not match experiment settings!"
            )

    def _get_group_sample(self, index, file_idx):
        all_data = []

        # count all samples from all previous files, correct index.
        nb_samples_prev_files = int(
            torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[:file_idx])
        )

        nb_datasets_in_group = self.nb_smpls_per_ds_per_grp_per_fn[file_idx][0].shape[0]

        for dataset_idx in range(nb_datasets_in_group):
            nb_samples_in_dataset = int(
                torch.sum(
                    self.nb_smpls_per_ds_per_grp_per_fn[file_idx][0][: dataset_idx + 1]
                )
            )

            if index < nb_samples_prev_files + nb_samples_in_dataset:
                break

        nb_samples_in_dataset = int(
            self.nb_smpls_per_ds_per_grp_per_fn[file_idx][0][dataset_idx]
        )

        # self.nb_smpls_per_ds_per_grp_per_fn.shape = [1,1,4]
        # self.nb_smpls_per_ds_per_grp_per_fn = [[[48, 48, 48, 48]]]
        # means: one file, one group, 4 datasets, each dataset with 48 samples

        if nb_samples_in_dataset == 1:
            sample_idx = 0
        else:
            offset_current_file = int(
                torch.sum(
                    self.nb_smpls_per_ds_per_grp_per_fn[file_idx][0][:dataset_idx]
                )
            )
            sample_idx = index - nb_samples_prev_files - offset_current_file

        for group_idx, req_group in enumerate(self.hdf_ds_key_request):
            ds = self.hdf_data_tensor[file_idx][group_idx][dataset_idx][sample_idx]
            transform = self.transformers[self.transform_names[group_idx]]
            if transform is None:
                all_data.append(ds)
            else:
                all_data.append(transform(ds))

        return tuple(all_data)

    def _get_dataset_sample(self, index, file_idx):
        raise NotImplementedError("This case is currently not supported! E2")
        # all_data = []

        # offset_file = int(torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[:file_idx]))
        # sample_idx = index - offset_file

        # for i, req_ds in enumerate(self.hdf_ds_key_request):
        #     ds = self.hdf_data_tensor[file_idx][0][i][sample_idx]
        #     transform = self._get_transform(req_ds)
        #     if transform is None:
        #         all_data.append(ds)
        #     else:
        #         all_data.append(transform(ds))

        # return tuple(all_data)

    def __getitem__(self, index):
        # Select file
        for file_idx in range(len(self.dataset_file_paths)):
            nb_samples_current_file = int(
                torch.sum(self.nb_smpls_per_ds_per_grp_per_fn[: file_idx + 1])
            )

            if index < nb_samples_current_file:
                break

        if self.groups_are_sample_pairs:
            sample = self._get_group_sample(index, file_idx)

        elif self.datasets_are_sample_pairs:
            sample = self._get_dataset_sample(index, file_idx)

        else:
            sample = self._get_sample(index, file_idx)

        return sample

    def __len__(self):
        return self.total_nb_samples


def create_datasets(experiment, args=None):
    """
    Data preparator for HDF5 files.

    Possible experimental setups:
    1) One directory with one or many files for all data (train, val, test).
    2) Sub-directories for each data set (train, val, test) with one or many files.

    Each file can contain one or many groups,
    Each group can contain one or many datasets,
    Each dataset can contain one or many samples.

    Available preparator settings:
    - dataset_workers (int): number of workers for data loading (defaults to 4)
    - dataset_transforms_train_a ([string]): list of transforms (as string) for training data
    - Resize_IMG_SIZE (int): resize images to IMG_SIZE (this is a parameter for the Resize transform)
    - subset_train (float): ratio of training data to use (defaults to 1.0)
    - subset_val (float): ratio of validation data to use (defaults to 1.0)
    - subset_test (float): ratio of test data to use (defaults to 1.0)
    - merge_test_train_val (bool): merge all data sets into one (defaults to False)
    - file_extension (string): file extension of data files (filter by type, defaults to "h5")
    - hdf_ds_key_request ([string]): which keys to load from hdf file
    - data_is_3d (bool): load data as 3D tensor (defaults to False)
    - TODO: dont store test set in ram during training.
    - TODO: dont store full data in ram if subset is used.
    """

    # def _cleanup_hdf_keys(requests):
    #     if requests is None:
    #         return None
    #     if not isinstance(requests, list):
    #         raise ValueError("hdf_ds_key_request must be [str,..] or [{str: str},..] !")
    #     if isinstance(requests[0], dict):
    #         return [k for d in args.hdf_ds_key_request for k in d.keys()]
    #     else:
    #         return args.hdf_ds_key_request

    # hdf_ds_key_request = _cleanup_hdf_keys(args.hdf_ds_key_request)
    # hdf_ds_key_request_test = _cleanup_hdf_keys(args.hdf_ds_key_request_test)
    hdf_ds_key_request = args.hdf_ds_key_request
    hdf_ds_key_request_test = args.hdf_ds_key_request_test
    if hdf_ds_key_request_test is None:
        hdf_ds_key_request_test = hdf_ds_key_request

    file_extension = args.get("file_extension", "hdf")

    data_files = {}

    data_basePath = args.base_path

    # use dataBasePath (searches for all files in this folder)
    if args.base_path is not None:
        if (
            args.train_files_path is not None
            or args.test_files_path is not None
            or args.val_files_path is not None
        ):
            raise ValueError(
                "This combination of data path definitions is not allowed!!"
            )

        # look for files in subdirectories
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

        # no specific train set defined -> use all data for training
        if len(data_files[DatasetType.Train]) == 0:
            if (
                len(data_files[DatasetType.Test]) != 0
                or len(data_files[DatasetType.Val]) != 0
            ):
                raise ValueError(
                    "No train set defined, but test or validation set! This case is currently not supported!"
                )
            data_files[DatasetType.Train] = data_files[DatasetType.All]

    # define files directly in form of a list of paths
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

    # load train & val only in the case of training
    is_train = experiment.mode.op_mode.train
    if is_train:
        iprint("Preparing training dataset")
        train = H5Dataset(
            file_paths=data_files[DatasetType.Train],
            experiment=experiment,
            hdf_ds_def=hdf_ds_key_request,
            args=args,
        )

        if len(data_files[DatasetType.Val]) > 0:
            iprint("Preparing validation dataset")
            val = H5Dataset(
                file_paths=data_files[DatasetType.Val],
                experiment=experiment,
                hdf_ds_def=hdf_ds_key_request,
                args=args,
            )
        else:
            val = []
    else:
        iprint(
            "This is a test run - no training and validation sets are being generated."
        )
        train = []
        val = []

    is_test = experiment.mode.op_mode.test
    if is_test and len(data_files[DatasetType.Test]) > 0:
        iprint("Preparing test dataset")
        test = H5Dataset(
            file_paths=data_files[DatasetType.Test],
            experiment=experiment,
            hdf_ds_def=hdf_ds_key_request_test,
            args=args,
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
