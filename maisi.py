import os
import torch
import tempfile
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.utils import download_url
from torch.nn.modules import Module
from fdq.ui_functions import startProgBar, iprint
from image_functions import createSubplots
from ae_test_eval import AETestEvaluator


# Pretrained MAISI VAE (Guo et al., "MAISI: Medical AI for Synthetic Imaging", WACV 2025,
# https://arxiv.org/abs/2409.11169), downloaded from NVIDIA's official location as listed in
# the MONAI model zoo bundle "maisi_ct_generative" (large_files.yml):
# https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative
# The weights are NOT part of this repository and are subject to NVIDIA's model weights
# license (non-commercial, research/evaluation use only), see the bundle's LICENSE file.
MAISI_URL = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"

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
        weights_path = os.path.join(tmpdir, "weights.pt")
        download_url(url, filepath=weights_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
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

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)

    evaluator = AETestEvaluator(experiment, model_name="MAISI")

    targs = experiment.cfg.train.args
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

        pbar.update(nb_tbatch)

        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            if isinstance(batch, (list, tuple)):
                # Legacy loader compatibility: TODO cleanup!
                images_gt = batch[0].to(experiment.device)
            else:
                images_gt = batch.to(experiment.device)
            z_mu, z_sigma = maisi_model.encode(images_gt)
            z_vae = maisi_model.sampling(z_mu, z_sigma)
            z_vae_decoded = maisi_model.decode(z_vae)

        latent_imgs_path = createSubplots(
            image_list=z_vae[:, :3, ...],
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title=f"Latent sample (CH1-3). orig shape: {z_vae.shape}",
            export_transform=evaluator.t_axial,
            hide_ticks=True,
        )

        evaluator.add_sample(
            nb_tbatch,
            images_gt,
            z_vae_decoded,
            extra_images=[{"name": "latent", "path": latent_imgs_path}],
        )

    pbar.finish()

    return evaluator.finish()
