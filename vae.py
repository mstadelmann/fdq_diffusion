import torch
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb
from image_functions import createSubplots

# from monai.losses.perceptual import PerceptualLoss


def KL_loss(z_mu, z_sigma):
    """
    Compute the Kullback-Leibler (KL) divergence loss for a variational autoencoder (VAE).

    The KL divergence measures how one probability distribution diverges from a second, expected probability distribution.
    In the context of VAEs, this loss term ensures that the learned latent space distribution is close to a standard normal distribution.

    Args:
        z_mu (torch.Tensor): Mean of the latent variable distribution, shape [N,C,H,W,D] or [N,C,H,W].
        z_sigma (torch.Tensor): Standard deviation of the latent variable distribution, same shape as 'z_mu'.

    Returns:
        torch.Tensor: The computed KL divergence loss, averaged over the batch.
    """
    eps = 1e-10
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def fdq_train(experiment) -> None:
    iprint("Chuchichaestli Diffusion Training")

    exp_trans_name = experiment.exp_def.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]

    train_loader = data.train_data_loader

    # loss_perceptual = (
    #     PerceptualLoss(
    #         spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
    #     )
    #     .eval()
    #     .to(experiment.device)
    # )

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        model.train()
        train_loss_sum = 0.0
        train_kl_loss_sum = 0.0
        train_recon_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")


        if experiment.is_distributed():
            # necessary to make shuffling work properly
            data.train_sampler.set_epoch(epoch)
            data.val_sampler.set_epoch(epoch)

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)

            is_grayscale = images_gt.shape[1] == 1

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                reconstruction, z_mu, z_sigma = model(images_gt)

                recon_loss = experiment.losses["MAE"](images_gt, reconstruction)
                kl_loss = KL_loss(z_mu, z_sigma)
                train_loss_tensor = (
                    recon_loss + targs.kl_loss_weight * kl_loss
                ) / experiment.gradacc_iter

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
            train_kl_loss_sum += kl_loss.detach().item()
            train_recon_loss_sum += recon_loss.detach().item()

        pbar.finish()
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        train_kl_loss_sum = train_kl_loss_sum / len(data.val_data_loader.dataset)
        train_recon_loss_sum = train_recon_loss_sum / len(data.val_data_loader.dataset)

        model.eval()
        pbar = startProgBar(data.n_val_batches, "validation...")
        val_loss_sum = 0.0
        val_kl_loss_sum = 0.0
        val_recon_loss_sum = 0.0

        with torch.no_grad():
            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch)

                images_gt = batch[0].to(experiment.device)

                reconstruction, z_mu, z_sigma = model(images_gt)
                # z_mu, z_sigma = model.encode(images_gt)
                # z_vae = model.sampling(z_mu, z_sigma)
                # reconstruction = model.decode(z_vae)

                recon_loss = experiment.losses["MAE"](images_gt, reconstruction)
                kl_loss = KL_loss(z_mu, z_sigma)

                val_loss_tensor = recon_loss + targs.kl_loss_weight * kl_loss

                val_loss_sum += val_loss_tensor.detach().item()
                val_recon_loss_sum += recon_loss.detach().item()
                val_kl_loss_sum += kl_loss.detach().item()

            experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)
            val_recon_loss_sum = val_recon_loss_sum / len(data.val_data_loader.dataset)
            val_kl_loss_sum = val_kl_loss_sum / len(data.val_data_loader.dataset)
            pbar.finish()

            # epsilon = torch.randn_like(z_sigma)
            # z_sample = z_mu + z_sigma * epsilon
            nb_imgs = min(
                experiment.exp_def.store.get("img_exp_nb", 4), images_gt.shape[0]
            )

            log_scalars = {
                "train_recon_loss": train_recon_loss_sum,
                "train_kl_loss": train_kl_loss_sum,
                "val_recon_loss": val_recon_loss_sum,
                "val_kl_loss": val_kl_loss_sum,
            }

            val_gt_path = sigma_histo_path = createSubplots(
                image_list=images_gt[:nb_imgs, ...],
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="val_gt",
                hide_ticks=True,
                show_colorbar=False,
                export_transform=t_img_exp,
            )

            mu_histo_path = createSubplots(
                image_list=z_mu[:nb_imgs, ...],
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="mu",
                hide_ticks=True,
                show_colorbar=False,
                export_transform=t_img_exp,
            )

            sigma_histo_path = createSubplots(
                image_list=z_sigma[:nb_imgs, ...],
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="sigma",
                hide_ticks=True,
                show_colorbar=False,
                export_transform=t_img_exp,
            )

            val_recon_path = sigma_histo_path = createSubplots(
                image_list=reconstruction[:nb_imgs, ...],
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="val_recon",
                hide_ticks=True,
                show_colorbar=False,
                export_transform=t_img_exp,
            )

            diff_path = sigma_histo_path = createSubplots(
                image_list=torch.abs(images_gt - reconstruction)[:nb_imgs, ...],
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="diff: abs(val_gt - val_recon)",
                hide_ticks=True,
                show_colorbar=False,
                export_transform=t_img_exp,
            )

            imgs_to_log = [
                {"name": "val_gt_histo", "path": val_gt_path},
                {"name": "val_recon_histo", "path": val_recon_path},
                {"name": "diff abs(gt - recon)", "path": diff_path},
                {"name": "val_mu_h", "path": mu_histo_path},
                {"name": "val_sigma_h", "path": sigma_histo_path},
                # {"name": "val_sample", "data": z_sample[:nb_imgs, ...]},
            ]

        experiment.finalize_epoch(log_scalars=log_scalars, log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break


@torch.no_grad()
def fdq_test(experiment):
    """Basic VAE evaluation."""

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]

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

    test_loader = data.test_data_loader
    nb_test_samples = experiment.exp_def.test.args.get("nb_test_samples", 10)

    if experiment.exp_def.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
        )

    kl_losses = []
    recon_losses = []
    weighted_losses = []
    results = []

    print(f"Testset sample size: {data.n_test_samples}")
    pbar = startProgBar(data.n_test_samples, "evaluation...")

    for nb_tbatch, batch in enumerate(test_loader):
        if nb_tbatch >= nb_test_samples:
            break

        pbar.update(nb_tbatch)
        images_gt = batch[0].to(experiment.device)
        reconstruction, z_mu, z_sigma = model(images_gt)

        results.append(reconstruction.cpu())

        is_grayscale = images_gt.shape[1] == 1

        img_list = [
            images_gt,
            reconstruction,
            images_gt - reconstruction,
        ]

        labels = ["gt", "recon", "diff"]

        if norm_to_hu is not None:
            result_hu = norm_to_hu(reconstruction)
            images_gt_hu = norm_to_hu(images_gt)
            img_list.extend([images_gt_hu, result_hu])
            labels.extend(["gt_hu", "recon_hu"])
            PSNR = experiment.losses["PSNR"](images_gt, reconstruction).item()
            PSNR_hu = experiment.losses["PSNR_hu"](images_gt_hu, result_hu).item()
            log_scalars = {"PSNR_test": PSNR, "PSNR_test_HU": PSNR_hu}
        else:
            log_scalars = None

        eval_path = createSubplots(
            image_list=img_list,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title=f"test image {nb_tbatch}",
            labels=labels,
            export_transform=t_img_exp,
            show_colorbar=False,
            apply_global_range=False,
        )

        save_wandb(
            experiment,
            images=[
                {"name": "test_samples", "path": eval_path},
            ],
            scalars=log_scalars,
        )

        kl_loss = KL_loss(z_mu, z_sigma).detach().item()
        kl_losses.append(kl_loss)
        recon_loss = experiment.losses["MAE"](images_gt, reconstruction).detach().item()
        recon_losses.append(recon_loss)
        weighted_loss = (
            recon_loss + experiment.exp_def.train.args.kl_loss_weight * kl_loss
        )
        weighted_losses.append(weighted_loss)

    pbar.finish()

    results_scalars = {
        "weighted_loss": float(torch.tensor(weighted_losses).mean()),
        "recon_loss": float(torch.tensor(recon_losses).mean()),
        "kl_loss": float(torch.tensor(kl_losses).mean()),
    }

    print(results_scalars)

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

    images = [{"name": "results overview", "path": res_path}]

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
        images.append({"name": "results overview sag", "path": res_path_sag})

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
        images.append({"name": "results overview cor", "path": res_path_cor})

    save_wandb(experiment, images=images)

    return results_scalars
