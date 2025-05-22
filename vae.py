import torch
from torchvision import transforms
from fdq.misc import print_nb_weights
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots, get_norm_to_rgb

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


def train(experiment) -> None:

    iprint("Chuchichaestli Diffusion Training")
    print_nb_weights(experiment)

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    norm_to_rgb = get_norm_to_rgb(experiment)

    data = experiment.data["celeb_HDF"]
    model = experiment.models["monaivae"]
    targs = experiment.exp_def.train.args

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
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)

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
                    b_idx=nb_tbatch, loader_name="celeb_HDF", model_name="monaivae"
                )

            train_loss_sum += train_loss_tensor.detach().item()

        pbar.finish()
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation train_loss_tensor

        with torch.no_grad():
            epsilon = torch.randn_like(z_sigma)
            z_sample = z_mu + z_sigma * epsilon

            nbi = min(experiment.exp_def.store.get("img_exp_nb", 4), images_gt.shape[0])

            mu_histo_path = createSubplots(
                image_list=[img.detach().float() for img in z_mu[:nbi, ...]],
                grayscale=False,
                experiment=experiment,
                histogram=True,
                figure_title="mu",
            )

            sigma_histo_path = createSubplots(
                image_list=[img.detach().float() for img in z_sigma[:nbi, ...]],
                grayscale=False,
                experiment=experiment,
                histogram=True,
                figure_title="sigma",
            )

            imgs_to_log = [
                {"name": "input", "data": images_gt[:nbi, ...]},
                {"name": "recon", "data": reconstruction[:nbi, ...]},
                {
                    "name": "diff",
                    "data": torch.abs(images_gt - reconstruction)[:nbi, ...],
                },
                {"name": "sample", "data": z_sample[:nbi, ...]},
                {"name": "mu_h", "path": mu_histo_path},
                {"name": "sigma_h", "path": sigma_histo_path},
                {"name": "mu", "data": z_mu[:nbi, ...]},
                {"name": "sigma", "data": z_sigma[:nbi, ...]},
            ]

        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break
