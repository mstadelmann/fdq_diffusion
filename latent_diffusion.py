import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots


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
    vae_model = experiment.models[targs.encoder_name]

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
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch.to(experiment.device)
            z_mu, z_sigma = vae_model.encode(images_gt)
            z_vae = vae_model.sampling(z_mu, z_sigma)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                img_shape = z_vae.shape[1:]
                is_grayscale = images_gt.shape[1] == 1

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
                    image_list=z_vae,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input Latent",
                    export_transform=t_img_exp,
                    hide_ticks=True,
                    show_colorbar=False,
                )
                decoded_imgs_path = createSubplots(
                    image_list=vae_model.decode(z_vae),
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

        history_path_latent = createSubplots(
            image_list=diff_history_latent,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
        )

        diff_history_norm = [vae_model.decode(i) for i in diff_history_latent]

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
    unet_model = experiment.models[targs.model_name]
    vae_model = experiment.models[targs.encoder_name]
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

    test_sample = next(iter(data.test_data_loader))
    # Legacy loader compatibility: TODO cleanup!
    if isinstance(test_sample, (list, tuple)):
        test_sample = test_sample[0]
    test_sample = test_sample.to(experiment.device)

    vae_model.eval()
    _, z_sigma = vae_model.encode(test_sample)
    img_shape = z_sigma.shape[1:]
    is_grayscale = test_sample.shape[1] == 1

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

        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            diff_history_latent = get_sample_from_noise(
                model=unet_model,
                diffuser=chuchi_diffuser,
                gen_shape=img_shape,
                idx_to_store=idx_to_store,
            )
            diff_history_norm = [vae_model.decode(i) for i in diff_history_latent]

        history_path_latent = createSubplots(
            image_list=diff_history_latent,
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

        save_wandb(
            experiment,
            images=[
                {"name": "test_gen_history_lat", "path": history_path_latent},
                {"name": "test_gen_history", "path": history_path},
            ],
        )

        results.append(diff_history_norm[-1].cpu())

    pbar.finish()

    res_path = createSubplots(
        image_list=results,
        grayscale=is_grayscale,
        experiment=experiment,
        histogram=False,
        figure_title="Generated test samples",
        export_transform=t_img_exp,
        hide_ticks=True,
        show_colorbar=False,
    )

    images = [{"name": "test_samples", "path": res_path}]

    # sagital and coronal planes
    if "extract_2d_from_3d_sag" in experiment.transformers:
        res_path_sag = createSubplots(
            image_list=results,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=False,
            figure_title="Generated test samples sag",
            export_transform=experiment.transformers["extract_2d_from_3d_sag"],
            hide_ticks=True,
            show_colorbar=False,
        )
        images.append({"name": "test_samples sag", "path": res_path_sag})

    if "extract_2d_from_3d_cor" in experiment.transformers:
        res_path_cor = createSubplots(
            image_list=results,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=False,
            figure_title="Generated test samples cor",
            export_transform=experiment.transformers["extract_2d_from_3d_cor"],
            hide_ticks=True,
            show_colorbar=False,
        )
        images.append({"name": "test_samples cor", "path": res_path_cor})

    save_wandb(experiment, images=images)
