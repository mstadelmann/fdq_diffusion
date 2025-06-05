import torch
from torchvision import transforms
from fdq.ui_functions import startProgBar, iprint
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

    img_exp_op = experiment.exp_def.store.img_exp_transform
    if img_exp_op is None:
        t_img_exp = transforms.Lambda(lambda t: t)
    else:
        t_img_exp = experiment.transformers[img_exp_op]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    targs = experiment.exp_def.train.args
    is_3d = experiment.exp_def.data.get(targs.dataloader_name).args.data_is_3d
    is_grayscale = False

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

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                img_shape = images_gt.shape[1:]
                if img_shape[1] == 1:
                    is_grayscale = True

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

            epsilon = torch.randn_like(z_sigma)
            z_sample = z_mu + z_sigma * epsilon
            nb_imgs = min(
                experiment.exp_def.store.get("img_exp_nb", 4), images_gt.shape[0]
            )

            log_scalars = {
                "train_recon_loss": train_recon_loss_sum,
                "train_kl_loss": train_kl_loss_sum,
                "val_recon_loss": val_recon_loss_sum,
                "val_kl_loss": val_kl_loss_sum,
            }

            # if is_3d and z_mu.dim() == 5:
            #     mid_slice = z_mu.shape[2] // 2
            #     z_mu = z_mu[:, :, mid_slice, ...]
            #     z_sigma = z_sigma[:, :, mid_slice, ...]
            #     z_sample = z_sample[:, :, mid_slice, ...]
            #     images_gt = images_gt[:, :, mid_slice, ...]
            #     reconstruction = reconstruction[:, :, mid_slice, ...]

            mu_histo_path = createSubplots(
                image_list=t_img_exp(z_mu[:nb_imgs, ...].detach().float()),
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="mu",
            )

            sigma_histo_path = createSubplots(
                image_list=t_img_exp(z_sigma[:nb_imgs, ...].detach().float()),
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="sigma",
            )

            imgs_to_log = [
                {"name": "val_gt", "data": t_img_exp(images_gt[:nb_imgs, ...])},
                {"name": "val_recon", "data": t_img_exp(reconstruction[:nb_imgs, ...])},
                {
                    "name": "val_diff",
                    "data": t_img_exp(
                        torch.abs(images_gt - reconstruction)[:nb_imgs, ...]
                    ),
                },
                {"name": "val_sample", "data": t_img_exp(z_sample[:nb_imgs, ...])},
                {"name": "val_mu_h", "path": mu_histo_path},
                {"name": "val_sigma_h", "path": sigma_histo_path},
                {"name": "val_mu", "data": t_img_exp(z_mu[:nb_imgs, ...])},
                {"name": "val_sigma", "data": t_img_exp(z_sigma[:nb_imgs, ...])},
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

    img_exp_op = experiment.exp_def.store.img_exp_transform
    if img_exp_op is None:
        t_img_exp = transforms.Lambda(lambda t: t)
    else:
        t_img_exp = experiment.transformers[img_exp_op]

    test_loader = data.test_data_loader
    nb_test_samples = experiment.exp_def.test.args.get("nb_test_samples", 10)
    nb_export_samples = experiment.exp_def.test.args.get("nb_export_samples", 0)

    if experiment.exp_def.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
        )

    kl_losses = []
    recon_losses = []
    weighted_losses = []

    print(f"Testset sample size: {data.n_test_samples}")
    pbar = startProgBar(data.n_test_samples, "evaluation...")

    for nb_tbatch, batch in enumerate(test_loader):
        if nb_tbatch >= nb_test_samples:
            break

        pbar.update(nb_tbatch)
        images_gt = batch[0].to(experiment.device)
        reconstruction, z_mu, z_sigma = model(images_gt)

        if nb_tbatch <= nb_export_samples:
            img_list = [
                t_img_exp(images_gt),
                t_img_exp(reconstruction),
                t_img_exp(images_gt - reconstruction),
            ]

            createSubplots(
                image_list=img_list,
                grayscale=False,
                experiment=experiment,
                histogram=True,
                figure_title=f"test image {nb_tbatch}",
                labels=["gt", "recon", "diff"],
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

    results = {
        "weighted_loss": float(torch.tensor(weighted_losses).mean()),
        "recon_loss": float(torch.tensor(recon_losses).mean()),
        "kl_loss": float(torch.tensor(kl_losses).mean()),
    }

    print(results)

    return results
