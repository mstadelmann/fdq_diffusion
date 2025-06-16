import os
import torch
import tempfile
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.utils import download_url
from safetensors.torch import load_model
from torch.nn.modules import Module
from fdq.ui_functions import startProgBar, iprint
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

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    maisi_model = load_url(
        AutoencoderKlMaisi(**MAISI_ARGS), url=MAISI_URL, freeze=True
    ).to(experiment.device)

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    pbar = startProgBar(data.n_train_batches, "training...")

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

        gt_imgs_path = createSubplots(
            image_list=images_gt,
            grayscale=True,
            experiment=experiment,
            histogram=True,
            figure_title="Input GT",
            export_transform=t_img_exp,
        )
        latent_imgs_path = createSubplots(
            image_list=z_vae[:, :3, ...],
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title="Latent sample (CH1-3)",
            export_transform=t_img_exp,
        )

        decoded_imgs_path = createSubplots(
            image_list=z_vae_decoded,
            grayscale=True,
            experiment=experiment,
            histogram=True,
            figure_title="Decoded Latent",
            export_transform=t_img_exp,
        )
        diff_imgs_path = createSubplots(
            image_list=torch.abs(images_gt - z_vae_decoded),
            grayscale=True,
            experiment=experiment,
            histogram=True,
            figure_title="Diff: Input - Decoded Latent",
            export_transform=t_img_exp,
        )

        imgs_to_log.extend(
            [
                {"name": "gt", "path": gt_imgs_path},
                {"name": "latent", "path": latent_imgs_path},
                {
                    "name": "decoded",
                    "path": decoded_imgs_path,
                },
                {
                    "name": "diff",
                    "path": diff_imgs_path,
                },
            ]
        )

        pbar.finish()

        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break
