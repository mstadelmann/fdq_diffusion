import os
import torch
from torchvision import transforms
import tempfile
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.utils import download_url
from safetensors.torch import load_model
from torch.nn.modules import Module
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
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


@torch.no_grad()
def get_sample_from_noise(model, diffuser, gen_shape, idx_to_store=None):
    model.eval()

    imgs = []

    for j, img in enumerate(
        diffuser.generate(
            model=model,
            shape=tuple(gen_shape),
            n=1,
            yield_intermediate=True,
        )
    ):
        if idx_to_store is not None:
            if j in idx_to_store:
                imgs.append(img)
        else:
            imgs = img

    return imgs


def fdq_train(experiment) -> None:
    iprint("Chuchichaestli MAISI Diffusion Training")

    img_exp_op = experiment.exp_def.store.img_exp_transform
    if img_exp_op is None:
        t_img_exp = transforms.Lambda(lambda t: t)
    else:
        t_img_exp = experiment.transformers[img_exp_op]

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    unet_model = experiment.models[targs.model_name]
    maisi_model = load_url(
        AutoencoderKlMaisi(**MAISI_ARGS), url=MAISI_URL, freeze=True
    ).to(experiment.device)

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    train_loader = data.train_data_loader

    chuchi_diffuser = DDPM(
        num_timesteps=targs.diffusion_nb_steps,
        device=experiment.device,
        beta_start=targs.diffusion_shd_beta_start,
        beta_end=targs.diffusion_shd_beta_end,
        schedule=targs.diffusion_scheduler,
    )
    img_shape = None

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")
        imgs_to_log = []

        unet_model.train()
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)
            z_mu, z_sigma = maisi_model.encode(images_gt)
            z_vae = maisi_model.sampling(z_mu, z_sigma)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                img_shape = z_vae.shape[1:]

                # first test batch: store inputs
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
                    figure_title="Input Latent (CH1-3)",
                    export_transform=t_img_exp,
                )
                z_vae_decoded = maisi_model.decode(z_vae)
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
                        {"name": "train_in_imgs", "path": gt_imgs_path},
                        {"name": "train_latent_imgs", "path": latent_imgs_path},
                        {
                            "name": "train_in_lat_dec",
                            "path": decoded_imgs_path,
                        },
                        {
                            "name": "train_in_gt_decode_diff",
                            "path": diff_imgs_path,
                        },
                    ]
                )

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(z_vae)
                noise_pred = unet_model(noisy_imgs, t)
                train_loss_tensor = (
                    experiment.losses["MAE"](noise, noise_pred)
                    / experiment.gradacc_iter
                )

                if experiment.useAMP:
                    experiment.scaler.scale(train_loss_tensor).backward()
                else:
                    train_loss_tensor.backward()

                experiment.update_gradients(
                    b_idx=nb_tbatch,
                    loader_name=targs.dataloader_name,
                    model_name=targs.model_name,
                )

            train_loss_sum += train_loss_tensor.detach().item()

        pbar.finish()
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation train_loss_tensor

        # Dummy validation: generate samples
        idx_to_store = torch.linspace(
            0,
            targs.diffusion_nb_steps - 1,
            targs.get("diffusion_nb_plot_steps", 15),
            dtype=torch.int,
        ).tolist()

        diff_history_latent = get_sample_from_noise(
            model=unet_model,
            diffuser=chuchi_diffuser,
            gen_shape=img_shape,
            idx_to_store=idx_to_store,
        )

        diff_history_latent_2dimg = [
            t_img_exp(i[:, :3, ...]) for i in diff_history_latent
        ]

        history_path_latent = createSubplots(
            image_list=diff_history_latent_2dimg,
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            labels=[f"Step {i}" for i in idx_to_store],
        )

        diff_history_norm = [maisi_model.decode(i) for i in diff_history_latent]

        history_path = createSubplots(
            image_list=diff_history_norm,
            grayscale=True,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Pixel",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
        )

        imgs_to_log.extend(
            [
                {"name": "gen_result", "data": t_img_exp(diff_history_norm[-1])},
                {"name": "gen_hist_lat", "path": history_path_latent},
                {"name": "gen_hist", "path": history_path},
            ]
        )

        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break
