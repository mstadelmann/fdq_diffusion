import torch
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb
from image_functions import createSubplots
from ae_test_eval import AETestEvaluator

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

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    targs = experiment.cfg.train.args
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
        experiment.on_epoch_start(epoch=epoch)

        model.train()
        train_loss_sum = 0.0
        train_kl_loss_sum = 0.0
        train_recon_loss_sum = 0.0
        n_train_processed = 0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            if isinstance(batch, (list, tuple)):
                # Legacy loader compatibility: TODO cleanup!
                images_gt = batch[0].to(experiment.device)
            else:
                images_gt = batch.to(experiment.device)

            n_train_processed += images_gt.shape[0]
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
        experiment.trainLoss = train_loss_sum / n_train_processed
        train_kl_loss_sum = train_kl_loss_sum / n_train_processed
        train_recon_loss_sum = train_recon_loss_sum / n_train_processed

        model.eval()
        pbar = startProgBar(data.n_val_batches, "validation...")
        val_loss_sum = 0.0
        val_kl_loss_sum = 0.0
        val_recon_loss_sum = 0.0
        n_val_processed = 0

        with torch.no_grad():
            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch)

                if isinstance(batch, (list, tuple)):
                    # Legacy loader compatibility: TODO cleanup!
                    images_gt = batch[0].to(experiment.device)
                else:
                    images_gt = batch.to(experiment.device)

                n_val_processed += images_gt.shape[0]
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

            experiment.valLoss = val_loss_sum / n_val_processed
            val_recon_loss_sum = val_recon_loss_sum / n_val_processed
            val_kl_loss_sum = val_kl_loss_sum / n_val_processed
            pbar.finish()

            # epsilon = torch.randn_like(z_sigma)
            # z_sample = z_mu + z_sigma * epsilon
            nb_imgs = min(
                experiment.cfg.store.get("img_exp_nb", 4), images_gt.shape[0]
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

        experiment.on_epoch_end(log_scalars=log_scalars, log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break


@torch.no_grad()
def fdq_test(experiment):
    """Basic VAE evaluation."""

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)

    if experiment.cfg.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
        )

    evaluator = AETestEvaluator(experiment, model_name="VAE")

    print(f"Testset sample size: {data.n_test_samples}")
    pbar = startProgBar(nb_test_samples, "evaluation...")

    for nb_tbatch, batch in enumerate(data.test_data_loader):
        if nb_tbatch >= nb_test_samples:
            break

        pbar.update(nb_tbatch)
        if isinstance(batch, (list, tuple)):
            # Legacy loader compatibility: TODO cleanup!
            images_gt = batch[0].to(experiment.device)
        else:
            images_gt = batch.to(experiment.device)

        reconstruction, z_mu, z_sigma = model(images_gt)

        kl_loss = KL_loss(z_mu, z_sigma).detach().item()
        recon_loss = experiment.losses["MAE"](images_gt, reconstruction).detach().item()
        weighted_loss = recon_loss + targs.kl_loss_weight * kl_loss

        evaluator.add_sample(
            nb_tbatch,
            images_gt,
            reconstruction,
            extra_scalars={
                "weighted_loss": weighted_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
            },
        )

    pbar.finish()

    return evaluator.finish()
