import os
import torch
import tempfile
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.utils import download_url
from safetensors.torch import load_model
from torch.nn.modules import Module
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb
from image_functions import createSubplots


MAISI_URL = "https://drive.switch.ch/index.php/s/HXdr8gRS0GkajyS/download?path=%2F&files=maisai_autoencoder.safetensors"

MAISI_ARGS = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "num_channels": [64, 128, 256],
    "num_res_blocks": [2, 2, 2],
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "attention_levels": [False, False, False],
    "with_encoder_nonlocal_attn": False,
    "with_decoder_nonlocal_attn": False,
    "use_checkpointing": False,
    "use_convtranspose": False,
    "norm_float16": False,
    "num_splits": 4,
    "dim_split": 1,
}


def load_url(model: Module, url: str, freeze: bool = False) -> Module:
    """Loads a model from a URL, optionally freezing its parameters.
    Downloads the model weights from the given URL, loads them into the provided PyTorch module,
    and optionally freezes the model's parameters to prevent further training.
    Args:
        model (Module): The PyTorch module to load the weights into.
        url (str): The URL where the model weights are located.
        freeze (bool, optional): Whether to freeze the model's parameters. Defaults to False.
    Returns:
        Module: The PyTorch module with the loaded weights.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "weights.safetensors")
        download_url(url, filepath=weights_path)
        load_model(model, weights_path)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def fdq_train(experiment) -> None:
    raise NotImplementedError(
        "Training is not implemented for MAISI - This is a test-only script."
    )


@torch.no_grad()
def fdq_test(experiment):
    iprint("MAISI Encode/Decode Test")

    nb_test_samples = experiment.exp_def.test.args.get("nb_test_samples", 10)

    exp_trans_name = experiment.exp_def.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    if "norm_to_HU" in experiment.transformers:
        norm_to_hu = experiment.transformers["norm_to_HU"]
    else:
        norm_to_hu = None

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    maisi_model = load_url(
        AutoencoderKlMaisi(**MAISI_ARGS), url=MAISI_URL, freeze=True
    ).to(experiment.device)

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    pbar = startProgBar(nb_test_samples, "MAISI inference...")

    for nb_tbatch, batch in enumerate(data.test_data_loader):

        if nb_tbatch >= nb_test_samples:
            iprint(f"Stopping after {nb_test_samples} test samples.")
            break

        imgs_to_log = []

        pbar.update(nb_tbatch)

        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            images_gt = batch[0].to(experiment.device)
            z_mu, z_sigma = maisi_model.encode(images_gt)
            z_vae = maisi_model.sampling(z_mu, z_sigma)
            z_vae_decoded = maisi_model.decode(z_vae)

        img_list = [
            images_gt,
            z_vae_decoded,
            torch.abs(images_gt - z_vae_decoded),
        ]

        labels = ["GT", "Decoded", "abs(GT - Decoded)"]

        if norm_to_hu is not None:
            result_hu = norm_to_hu(z_vae_decoded)
            images_gt_hu = norm_to_hu(images_gt)
            img_list.extend([images_gt_hu, result_hu])
            labels.extend(["gt_hu", "recon_hu"])
            PSNR = experiment.losses["PSNR"](images_gt, z_vae_decoded).item()
            PSNR_hu = experiment.losses["PSNR_hu"](images_gt_hu, result_hu).item()
            log_scalars = {"PSNR_test": PSNR, "PSNR_test_HU": PSNR_hu}
        else:
            log_scalars = None

        gt_imgs_path = createSubplots(
            image_list=img_list,
            labels=labels,
            grayscale=True,
            experiment=experiment,
            histogram=True,
            figure_title="Maisi eval",
            hide_ticks=True,
            apply_global_range=False,
            export_transform=t_img_exp,
            show_colorbar=False,
        )
        latent_imgs_path = createSubplots(
            image_list=z_vae[:, :3, ...],
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title=f"Latent sample (CH1-3). orig shape: {z_vae.shape}",
            export_transform=t_img_exp,
            hide_ticks=True,
        )

        imgs_to_log.extend(
            [
                {"name": "inf_results", "path": gt_imgs_path},
                {"name": "latent", "path": latent_imgs_path},
            ]
        )

        save_wandb(experiment=experiment, images=imgs_to_log, scalars=log_scalars)

    pbar.finish()
