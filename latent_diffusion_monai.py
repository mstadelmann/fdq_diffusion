import torch
from fdq.ui_functions import startProgBar, iprint
from monai.inferers import DiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from image_functions import createSubplots


def fdq_train(experiment) -> None:
    iprint("Chuchichaestli Diffusion Training")

    exp_trans_name = experiment.exp_def.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    unet_model = experiment.models[targs.model_name]
    vae_model = experiment.models[targs.encoder_name]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    train_loader = data.train_data_loader

    supported_schedules = [
        "linear_beta",
        "scaled_linear_beta",
        "sigmoid_beta",
        "cosine",
    ]

    if (
        targs.diffusion_scheduler is not None
        and targs.diffusion_scheduler not in supported_schedules
    ):
        raise ValueError(
            f"Unsupported diffusion scheduler: {targs.diffusion_scheduler}. "
            f"Supported schedulers are: {supported_schedules}"
        )

    supported_scheduler_args = ["beta_start", "beta_end", "sig_range", "s"]

    scheduler_args = targs.get("diffusion_scheduler_args", {})
    if scheduler_args is None:
        scheduler_args = {}
    else:
        scheduler_args = scheduler_args.to_dict()
        for param in scheduler_args.keys():
            if param not in supported_scheduler_args:
                raise ValueError(
                    f"Unsupported scheduler argument: {param}. "
                    f"Supported arguments are: {supported_scheduler_args}"
                )

    scheduler = DDPMScheduler(
        num_train_timesteps=targs.diffusion_nb_steps,
        schedule=targs.diffusion_scheduler,
        **scheduler_args,
    )
    inferer = DiffusionInferer(scheduler)

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)
        imgs_to_log = []

        unet_model.train()
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)
            z_mu, z_sigma = vae_model.encode(images_gt)
            z_vae = vae_model.sampling(z_mu, z_sigma)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                # img_shape = z_vae.shape[1:]

                is_grayscale = images_gt.shape[1] == 1

                # first test batch: store inputs
                gt_imgs_path = createSubplots(
                    image_list=images_gt,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input GT",
                    hide_ticks=True,
                    show_colorbar=False,
                    export_transform=t_img_exp,
                )
                latent_imgs_path = createSubplots(
                    image_list=z_vae,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input Latent",
                    hide_ticks=True,
                    show_colorbar=False,
                    export_transform=t_img_exp,
                )
                decoded_imgs_path = createSubplots(
                    image_list=vae_model.decode(z_vae),
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Decoded Latent",
                    hide_ticks=True,
                    show_colorbar=False,
                    export_transform=t_img_exp,
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
                noise = torch.randn_like(z_vae).to(experiment.device)
                timesteps = torch.randint(
                    0,
                    targs.diffusion_nb_steps,
                    (z_vae.shape[0],),
                    device=z_vae.device,
                ).long()

                noise_pred = inferer(
                    inputs=z_vae,
                    diffusion_model=unet_model,
                    noise=noise,
                    timesteps=timesteps,
                )

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
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation train_loss_tensor

        # Dummy validation: generate samples
        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, enabled=experiment.useAMP),
        ):
            noise = torch.randn((1, *z_vae.shape[1:]), device=experiment.device)
            scheduler.set_timesteps(num_inference_steps=targs.diffusion_nb_steps)

            isteps = int(
                targs.diffusion_nb_steps / targs.get("diffusion_nb_plot_steps", 15)
            )

            image, intermediates = inferer.sample(
                input_noise=noise,
                diffusion_model=unet_model,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=isteps,
            )

        history_path_latent = createSubplots(
            image_list=intermediates,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            hide_ticks=True,
            show_colorbar=False,
            export_transform=t_img_exp,
        )

        diff_history_norm = [vae_model.decode(i) for i in intermediates]

        history_path = createSubplots(
            image_list=diff_history_norm,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Pixel",
            hide_ticks=True,
            show_colorbar=False,
            export_transform=t_img_exp,
        )

        imgs_to_log.extend(
            [
                {"name": "gen_result", "data": t_img_exp(image)},
                {"name": "gen_hist_lat", "path": history_path_latent},
                {"name": "gen_hist", "path": history_path},
            ]
        )

        experiment.on_epoch_end(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break
