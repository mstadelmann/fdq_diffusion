import torch
from torchvision import transforms
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
            z_mu, z_sigma = vae_model.encode(images_gt)
            z_vae = vae_model.sampling(z_mu, z_sigma)

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
                    grayscale=False,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input GT",
                    export_transform=t_img_exp,
                )
                latent_imgs_path = createSubplots(
                    image_list=z_vae,
                    grayscale=False,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input Latent",
                )
                decoded_imgs_path = createSubplots(
                    image_list=vae_model.decode(z_vae),
                    grayscale=False,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Decoded Latent",
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
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps - Latent",
            labels=[f"Step {i}" for i in idx_to_store],
        )

        diff_history_norm = [vae_model.decode(i) for i in diff_history_latent]

        history_path = createSubplots(
            image_list=diff_history_norm,
            grayscale=False,
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
