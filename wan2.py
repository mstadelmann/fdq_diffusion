import os
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from fdq.ui_functions import startProgBar, iprint
from image_functions import createSubplots
from ae_test_eval import AETestEvaluator


# TI2V-5B is the Wan2.2 variant that ships the new high-compression
# "Wan2.2-VAE" (4x16x16, https://github.com/Wan-Video/Wan2.2); the other
# Wan2.2 variants (T2V/I2V-A14B) reuse the original Wan2.1 VAE instead.
WAN22_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN22_VAE_SUBFOLDER = "vae"


def load_wan22_vae(freeze: bool = False) -> AutoencoderKLWan:
    """Loads the pretrained Wan2.2 video VAE from the Hugging Face mirror of
    https://github.com/Wan-Video/Wan2.2, optionally freezing its parameters.

    Args:
        freeze (bool, optional): Whether to freeze the model's parameters. Defaults to False.
    Returns:
        AutoencoderKLWan: The pretrained Wan2.2 VAE.
    """
    vae = AutoencoderKLWan.from_pretrained(
        WAN22_VAE_REPO,
        subfolder=WAN22_VAE_SUBFOLDER,
        torch_dtype=torch.float32,
        token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    if freeze:
        for param in vae.parameters():
            param.requires_grad = False
    return vae


def pad_depth_for_causal_vae(volume: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pads the volume's depth axis (fed to Wan as its temporal axis) up to
    the next frame count its causal encoder consumes in full.

    Wan's causal VAE processes frames as 1 initial frame plus groups of 4
    (1 + 4*k); any trailing frames past the last complete group are silently
    dropped by encode() instead of raising, so an un-padded 128-deep CBCT
    volume would lose its last 3 slices. Padding is undone after decoding.

    Args:
        volume (torch.Tensor): Input tensor of shape [N, C, D, H, W].
    Returns:
        tuple[torch.Tensor, int]: The (possibly) padded tensor and the depth
        actually consumed to reach it.
    """
    depth = volume.shape[2]
    target_depth = 1 + 4 * ((depth - 1 + 3) // 4)
    pad_amount = target_depth - depth
    if pad_amount == 0:
        return volume, depth
    return F.pad(volume, (0, 0, 0, 0, 0, pad_amount), mode="replicate"), depth


def fdq_train(experiment) -> None:
    raise NotImplementedError(
        "Training is not implemented for the Wan2.2 VAE - This is a test-only script."
    )


@torch.no_grad()
def fdq_test(experiment):
    iprint("Wan2.2 VAE Encode/Decode Test")

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)

    evaluator = AETestEvaluator(experiment, model_name="Wan2.2")

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    wan_vae = load_wan22_vae(freeze=True).to(experiment.device)

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    pbar = startProgBar(nb_test_samples, "Wan2.2 VAE inference...")

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

            # Wan2.2's VAE is a video autoencoder trained on 3-channel RGB
            # frames [N, 3, T, H, W]; the CBCT volume's depth axis stands in
            # for the temporal axis, and the single grayscale channel is
            # replicated 3x to match the pretrained encoder's input.
            images_rgb, orig_depth = pad_depth_for_causal_vae(
                images_gt.repeat(1, 3, 1, 1, 1)
            )
            posterior = wan_vae.encode(images_rgb).latent_dist
            z_vae = posterior.sample()
            decoded_rgb = wan_vae.decode(z_vae).sample[:, :, :orig_depth, ...]
            z_vae_decoded = decoded_rgb.mean(dim=1, keepdim=True)

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
