import argparse
import ast
import concurrent.futures
import multiprocessing
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def is_interleaved_volume(data: np.ndarray, threshold: float = 1.2) -> tuple[bool, float]:
    """
    Detects if a volume suffers from interleaved/mixed series corruption.
    Assumes data shape is (Z, Y, X).
    
    Returns:
        is_corrupted (bool): True if the volume is likely interleaved.
        ratio (float): The ratio of adjacent MAE to alternating MAE.
    """
    try:
        # Calculate Mean Absolute Error between adjacent slices (z and z+1)
        mae_adjacent = np.mean(np.abs(data[:-1, :, :] - data[1:, :, :]))
        
        # Calculate Mean Absolute Error between alternating slices (z and z+2)
        mae_alternating = np.mean(np.abs(data[:-2, :, :] - data[2:, :, :]))
        
        # Calculate the ratio with epsilon
        ratio = mae_adjacent / (mae_alternating + 1e-8)
        
        is_corrupted = ratio > threshold
        return is_corrupted, ratio
    except Exception as e:
        print(f"Error checking interleave: {e}")
        return False, 0.0


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [round(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
    return resized_array


def single_job(data_batched, split="train", overwrite=False, download_dir=None, output_base_dir=None, resolution=128):
    try:
        repo_id = "ibrahimhamamci/CT-RATE"
        directory_name = f"dataset/{split}/"
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        # path where raw data is downloaded and stored
        local_dir = download_dir

        # path where processed data is stored, e.g. <output_base_dir>/preproc_128/
        processed_dir = os.path.join(output_base_dir, f"preproc_{resolution}") + "/"

        # Resolution is the single knob: everything else is derived from it to
        # keep the same convention as the previously generated tiers (128/64/32).
        # - in-plane FOV is held constant at 384mm (128*3.0 = 64*6.0 = 32*12.0
        #   in the earlier tiers), so xy spacing = 384 / resolution.
        # - z depth continues the existing z_base = resolution/2 relationship.
        # - z spacing is set equal to xy spacing (isotropic voxels), giving a
        #   constant 192mm z-FOV across all resolutions. Note this differs from
        #   the earlier processed_128/64/32 tiers, whose z spacing wasn't a
        #   clean function of resolution (they used 5.125/7.539/19.78mm, an
        #   inconsistent z-FOV) - so a preproc_128 generated here will match
        #   the old processed_128 in-plane exactly, but not along z.
        target_xy_shape = [resolution, resolution]  # target shape in XY plane (x, y)
        target_z_base = resolution // 2
        hu_min, hu_max = -1000, 2000
        xy_spacing_mm = 384.0 / resolution
        target_spacing = [xy_spacing_mm, xy_spacing_mm, xy_spacing_mm]  # target spacing in mm (z, x, y)

        name = data_batched["VolumeName"].item()
        folder1 = name.split("_")[0]
        folder2 = name.split("_")[1]
        folder = folder1 + "_" + folder2
        folder3 = name.split("_")[2]
        subfolder = folder + "_" + folder3
        subfolder = directory_name + folder + "/" + subfolder

        # The output path only depends on names, not on the downloaded/processed
        # data, so it can be checked before doing any of the expensive work below.
        # This is what makes reruns with more samples (e.g. 1000 -> 1200) cheap:
        # already-processed volumes are skipped without a download or resample.
        nii_file = os.path.join(os.path.join(local_dir, subfolder), name)
        nii_file_new = nii_file.replace(local_dir + "dataset/", processed_dir)
        if not overwrite and os.path.exists(nii_file_new):
            print(f"Skipping (already exists): {nii_file_new}")
            return True

        print(subfolder)

        # Only hit the Hub if the raw file isn't already present locally.
        # hf_hub_download(local_dir=...) writes its own lock/bookkeeping files
        # under local_dir/.cache/huggingface/download/... - on a shared,
        # not-fully-writable project path (like this one) that fails with a
        # PermissionError for any subfolder that download step hasn't already
        # created, even though the actual data file is already there and
        # doesn't need re-downloading.
        if not os.path.exists(nii_file):
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
                subfolder=subfolder,
                filename=name,
                local_dir=local_dir,
                # force_download=True,
            )

        row = data_batched
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = np.array(ast.literal_eval(row["XYSpacing"].iloc[0]))
        z_spacing = float(row["ZSpacing"].iloc[0])
        orientation = ast.literal_eval(row["ImageOrientationPatient"].iloc[0])
        origin = ast.literal_eval(row["ImagePositionPatient"].iloc[0])

        print("Nifti File:", nii_file)

        os.makedirs(os.path.dirname(nii_file_new), exist_ok=True)

        ################################# Image ###################################
        orig_image = sitk.ReadImage(nii_file)

        # get the image array
        nii_data = sitk.GetArrayFromImage(orig_image)

        # ---------------------------------------------------------
        # FIX: Dynamically check if the image is already in HU.
        # If the minimum value is reasonably negative (e.g., air at -1000 
        # or Siemens padding at -8192), SimpleITK already applied the intercept.
        # If it is > -500, it is raw unsigned data and needs the CSV transform.
        # ---------------------------------------------------------
        if np.min(nii_data) > -500:
            nii_data = slope * nii_data + intercept
            
        # Optional but recommended: Clean up the extreme -8192 Siemens padding 
        # to standard air (-1000) BEFORE interpolation to prevent edge bleeding.
        nii_data[nii_data < hu_min] = hu_min

        # Clip to your HU window
        nii_data = np.clip(nii_data, a_min=hu_min, a_max=hu_max)

        # Reorient to RAS FIRST...
        ct_image = sitk.GetImageFromArray(nii_data)
        
        # ---------------------------------------------------------
        # FIX: Build the Direction Matrix strictly from the CSV 
        # to bypass corrupted NIfTI headers that scramble DICOMOrient.
        # ---------------------------------------------------------
        iop = np.array(orientation) # Parsed earlier from ImageOrientationPatient
        x_vec = iop[:3]
        y_vec = iop[3:]
        # The slice direction (Z) is the cross product of the row (X) and column (Y) vectors
        z_vec = np.cross(x_vec, y_vec)
        
        true_direction = (
            float(x_vec[0]), float(x_vec[1]), float(x_vec[2]),
            float(y_vec[0]), float(y_vec[1]), float(y_vec[2]),
            float(z_vec[0]), float(z_vec[1]), float(z_vec[2])
        )
        
        ct_image.SetDirection(true_direction)
        
        # Override Origin and Spacing with CSV metadata as well
        ct_image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
        ct_image.SetSpacing((float(xy_spacing[0]), float(xy_spacing[1]), float(z_spacing)))
        
        ct_image = sitk.DICOMOrient(ct_image, "RAS")

        # Now extract the RAS-aligned array and spacing
        arr_ras = sitk.GetArrayFromImage(ct_image) # shape is (Z, Y, X)
        ras_spacing = ct_image.GetSpacing()        # shape is (X, Y, Z)

        # Interleaved/mixed-series volumes can't be reliably split back into
        # their component series from pixel data alone, so these are just
        # logged and skipped rather than processed.
        corrupted, score = is_interleaved_volume(arr_ras)
        if corrupted:
            print(f"[{score:.2f}] WARNING: Interleaved volume detected in {name}! Skipping...")
            log_file = os.path.join(processed_dir, f"interleaved_volumes_{split}.txt")
            with open(log_file, "a") as f:
                f.write(f"{name}\n")
            return True

        # Map spacing as (Z, Y, X) to match the (Z, Y, X) dimension order of `arr_ras`.
        current_spacing = (ras_spacing[2], ras_spacing[1], ras_spacing[0])

        # Force the strict target spacing, regardless of the resulting number of slices
        new_target_spacing = [target_spacing[0], target_spacing[2], target_spacing[1]]

        # reshape to desired voxel spacing
        tensor_data = torch.tensor(arr_ras, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
        tensor_data = resize_array(tensor_data, current_spacing, new_target_spacing)
        tensor_data = tensor_data[0][0]

        # Crop volume if it has more slices than specified
        if tensor_data.shape[0] > target_z_base:
            slice_start = (tensor_data.shape[0] - target_z_base) // 2
            slice_end = slice_start + target_z_base
            tensor_data = tensor_data[slice_start:slice_end]

        elif tensor_data.shape[0] < target_z_base:
            # If fewer slices than target_z_base, pad by replicating the first and last slices
            pad_size = target_z_base - tensor_data.shape[0]
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before

            parts = []
            if pad_before > 0:
                parts.append(torch.stack([tensor_data[0]] * pad_before))
            parts.append(tensor_data)
            if pad_after > 0:
                parts.append(torch.stack([tensor_data[-1]] * pad_after))

            tensor_data = torch.cat(parts, dim=0)

        # center crop and pad XY plane
        _, h, w = tensor_data.shape
        dh, dw = target_xy_shape

        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)

        tensor_data = tensor_data[:, h_start:h_end, w_start:w_end]

        pad_w_before = (dw - tensor_data.shape[2]) // 2  # pad last dim (w)
        pad_w_after = dw - tensor_data.shape[2] - pad_w_before
        pad_h_before = (dh - tensor_data.shape[1]) // 2  # pad second to last dim (h)
        pad_h_after = dh - tensor_data.shape[1] - pad_h_before

        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
        tensor_data = torch.nn.functional.pad(
            tensor_data, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, 0, 0), value=-1000
        )
        tensor_data = tensor_data[0][0]

        # clip to value range
        tensor_data = tensor_data.clip(hu_min, hu_max)

        # save the nii file as int16 format, change header accordingly
        nii_data_new = tensor_data.numpy().astype('int16')

        # create sitk image from nii_data
        image = sitk.GetImageFromArray(nii_data_new)

        # SITK requires spacing ordered as (X, Y, Z)
        image.SetSpacing((new_target_spacing[2], new_target_spacing[1], new_target_spacing[0]))
        image.SetOrigin(ct_image.GetOrigin())
        image.SetDirection(ct_image.GetDirection())

        # cast the image to int16
        image = sitk.Cast(image, sitk.sitkInt16)

        # save the image
        sitk.WriteImage(image, nii_file_new)
        print("Saved NIfTI:", nii_file_new)

    except Exception as e:
        # Log the error and continue
        print(f"Error processing {data_batched['VolumeName'].item()}: {e}")
        return False  # Indicate failure for this job
    return True  # Indicate success for this job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CT-RATE dataset splits.")
    parser.add_argument("--split", type=str, default="train", help="Which dataset split to process (e.g., train, valid)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of volumes to process, in metadata-row order (default: all). "
        "Useful for a quick test run, e.g. --num-samples 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output files even if they already exist (default: skip existing files).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). Raise this once you have "
        "enough CPU/memory allocated - each worker holds a full volume in memory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cluster/projects/2026_stmd_orthosense/public_data/ct-rate",
        help="Base directory for processed output. Files for a given --resolution R are "
        "written under <output-dir>/preproc_R/.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/cluster/projects/ac3t/data/ac3t_ct_rate/CT-RATE/",
        help="Directory where the raw CT-RATE NIfTI files are (or will be) stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Output in-plane resolution R, producing RxR images (default: 128). "
        "z depth and voxel spacing are derived from this - see single_job().",
    )
    args = parser.parse_args()

    split = args.split
    meta_path = "/cluster/projects/ac3t/data/ac3t_ct_rate/dataset_stats/" + f"{split}_metadata.csv"  # TODO: give proper path
    start_at = 0  # TODO: start from which row, 0 means beginning

    data = pd.read_csv(meta_path)
    rows_list = [data.iloc[[i]] for i in range(start_at, len(data))]

    if args.num_samples is not None:
        rows_list = rows_list[: args.num_samples]

    # # for debugging purpose, run a few jobs sequentially instead of the full
    # # parallel pass below:
    # for i in range(len(rows_list)):
    #     print(f"Processing job {i + 1}/{len(rows_list)}: {rows_list[i]['VolumeName'].item()}")
    #     single_job(rows_list[i], split, args.overwrite, args.download_dir, args.output_dir, args.resolution)

    ################## Parallel Processing ##################
    multiprocessing.set_start_method("spawn", force=True)
    # Use a ProcessPoolExecutor to run the jobs in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                single_job, row, split, args.overwrite, args.download_dir, args.output_dir, args.resolution
            ): i
            for i, row in enumerate(rows_list)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows_list), desc="Processing jobs"):
            try:
                if not future.result():
                    print(f"Job {futures[future]} failed.")
            except Exception as e:
                print(f"Unexpected error in job {futures[future]}: {e}")
