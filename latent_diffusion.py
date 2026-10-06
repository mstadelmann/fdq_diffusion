import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots, stack_images_vertically


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


def get_encoder(experiment):
    """Frozen autoencoder that maps images to/from the diffusion latent space.

    train.args.encoder_name is either the name of a model in experiment.models (a
    VAE trained with vae.py, loaded via its trained_model_path), or "maisi" for the
    pretrained NVIDIA MAISI VAE, downloaded from MAISI_URL (see maisi.py).
    """
    encoder_name = experiment.cfg.train.args.encoder_name
    if encoder_name != "maisi":
        return experiment.models[encoder_name]
    # imported here so the non-MAISI configs don't need monai's maisi deps
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
        AutoencoderKlMaisi,
    )
    from maisi import MAISI_ARGS, MAISI_URL, load_url

    return load_url(AutoencoderKlMaisi(**MAISI_ARGS), url=MAISI_URL, freeze=True).to(
        experiment.device
    )


def get_images_gt(batch, device):
    # nifity_preparator returns a tuple when transform_train/test is a list
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch.to(device)


def latent_for_plot(z, nb_img_channels):
    # MAISI latents have 4 channels for a 1-channel CT: only plot the first one
    if z.shape[1] != nb_img_channels:
        return z[:, :1, ...]
    return z


def fdq_train(experiment) -> None:
    iprint("Chuchichaestli Diffusion Training")

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    unet_model = experiment.models[targs.model_name]
    vae_model = get_encoder(experiment)
    vae_model.eval()

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
        experiment.on_epoch_start(epoch=epoch)
        imgs_to_log = []

        unet_model.train()
        train_loss_sum = 0.0
        n_train_processed = 0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = get_images_gt(batch, experiment.device)
            n_train_processed += images_gt.shape[0]
            with torch.no_grad(), torch.autocast(
                device_type=device_type, enabled=experiment.useAMP
            ):
                z_mu, z_sigma = vae_model.encode(images_gt)
                z_vae = vae_model.sampling(z_mu, z_sigma)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                img_shape = z_vae.shape[1:]
                nb_img_channels = images_gt.shape[1]
                is_grayscale = nb_img_channels == 1

                # first test batch: store inputs
                gt_imgs_path = createSubplots(
                    image_list=images_gt,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input GT",
                    export_transform=t_img_exp,
                    hide_ticks=True,
                    show_colorbar=False,
                )
                latent_imgs_path = createSubplots(
                    image_list=latent_for_plot(z_vae, nb_img_channels),
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input Latent",
                    export_transform=t_img_exp,
                    hide_ticks=True,
                    show_colorbar=False,
                )
                with torch.no_grad():
                    z_vae_decoded = vae_model.decode(z_vae)
                decoded_imgs_path = createSubplots(
                    image_list=z_vae_decoded,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Decoded Latent",
                    export_transform=t_img_exp,
                    hide_ticks=True,
                    show_colorbar=False,
                )

                imgs_to_log.extend(
                    [
                        {"name": "train_in", "path": gt_imgs_path},
                        {"name": "train_in_lat", "path": latent_imgs_path},
                        {
                            "name": "train_in_lat_dec",
                            "path": decoded_imgs_path,
                        },
                    ]
                )

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(z_vae)
                noise_pred = unet_model(noisy_imgs, t)
                train_loss_tensor = (
                    experiment.losses["MSE"](noise, noise_pred)
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
        experiment.trainLoss = train_loss_sum / n_train_processed

        unet_model.eval()
        val_loss_sum = 0.0
        n_val_processed = 0
        with torch.no_grad():
            for batch in data.val_data_loader:
                images_gt = get_images_gt(batch, experiment.device)
                n_val_processed += images_gt.shape[0]
                with torch.autocast(
                    device_type=device_type, enabled=experiment.useAMP
                ):
                    z_mu, z_sigma = vae_model.encode(images_gt)
                    z_vae = vae_model.sampling(z_mu, z_sigma)
                    noisy_imgs, noise, t = chuchi_diffuser.noise_step(z_vae)
                    noise_pred = unet_model(noisy_imgs, t)
                    val_loss_sum += experiment.losses["MSE"](noise, noise_pred).item()
        experiment.valLoss = val_loss_sum / n_val_processed

        # generate samples for preview
        idx_to_store = torch.linspace(
            0,
            targs.diffusion_nb_steps - 1,
            targs.get("diffusion_nb_plot_steps", 15),
            dtype=torch.int,
        ).tolist()

        diff_history_latent = get_sample_from_noise(
            model=experiment.get_eval_model(targs.model_name),
            diffuser=chuchi_diffuser,
            gen_shape=img_shape,
            idx_to_store=idx_to_store,
        )

        history_path_latent = createSubplots(
            image_list=[latent_for_plot(i, nb_img_channels) for i in diff_history_latent],
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
        )

        with torch.no_grad(), torch.autocast(
            device_type=device_type, enabled=experiment.useAMP
        ):
            diff_history_norm = [vae_model.decode(i).float() for i in diff_history_latent]

        history_path = createSubplots(
            image_list=diff_history_norm,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Pixel",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
        )

        imgs_to_log.extend(
            [
                {"name": "gen_result", "data": t_img_exp(diff_history_norm[-1])},
                {"name": "gen_hist_lat", "path": history_path_latent},
                {"name": "gen_hist", "path": history_path},
            ]
        )

        experiment.on_epoch_end(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break


@torch.no_grad()
def fdq_test(experiment):
    # This test is only for generative latent diffusion!

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    vae_model = get_encoder(experiment)
    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    if experiment.cfg.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 - please change the experiment file."
        )

    idx_to_store = torch.linspace(
        0,
        targs.diffusion_nb_steps - 1,
        targs.get("diffusion_nb_plot_steps", 15),
        dtype=torch.int,
    ).tolist()

    test_sample = get_images_gt(next(iter(data.test_data_loader)), experiment.device)

    vae_model.eval()
    with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
        _, z_sigma = vae_model.encode(test_sample)
    img_shape = z_sigma.shape[1:]
    nb_img_channels = test_sample.shape[1]
    is_grayscale = nb_img_channels == 1

    chuchi_diffuser = DDPM(
        num_timesteps=targs.diffusion_nb_steps,
        device=experiment.device,
        beta_start=targs.diffusion_shd_beta_start,
        beta_end=targs.diffusion_shd_beta_end,
        schedule=targs.diffusion_scheduler,
    )

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)
    pbar = startProgBar(nb_test_samples, "evaluation...")

    results = []

    for inf_nb in range(nb_test_samples):
        pbar.update(inf_nb)

        diff_history_latent = get_sample_from_noise(
            model=experiment.get_eval_model(targs.model_name),
            diffuser=chuchi_diffuser,
            gen_shape=img_shape,
            idx_to_store=idx_to_store,
        )
        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            diff_history_norm = [
                vae_model.decode(i).float() for i in diff_history_latent
            ]

        history_path_latent = createSubplots(
            image_list=[latent_for_plot(i, nb_img_channels) for i in diff_history_latent],
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
        )

        history_path = createSubplots(
            image_list=diff_history_norm,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Pixel",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
        )

        # pixel history (2 rows) on top, latent history (2 rows) below
        combined_history_path = stack_images_vertically(
            [history_path, history_path_latent],
            save_path=experiment.get_next_export_fn(name="test_gen_history"),
        )
        save_wandb(
            experiment,
            images=[{"name": "test_gen_history", "path": combined_history_path}],
        )

        results.append(diff_history_norm[-1].cpu())

    pbar.finish()

    # one row per plane (transversal, sagittal, coronal), stacked into one image
    views = [("transversal", t_img_exp)]
    if "extract_2d_from_3d_sag" in experiment.transformers:
        views.append(("sagittal", experiment.transformers["extract_2d_from_3d_sag"]))
    if "extract_2d_from_3d_cor" in experiment.transformers:
        views.append(("coronal", experiment.transformers["extract_2d_from_3d_cor"]))

    view_paths = [
        createSubplots(
            image_list=results,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=False,
            figure_title=f"Generated test samples - {view_name}",
            max_cols=len(results),
            export_transform=view_transform,
            hide_ticks=True,
            show_colorbar=False,
        )
        for view_name, view_transform in views
    ]
    res_path = stack_images_vertically(
        view_paths, save_path=experiment.get_next_export_fn(name="test_samples")
    )

    save_wandb(experiment, images=[{"name": "test_samples", "path": res_path}])
