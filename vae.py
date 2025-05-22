import torch
from torchvision import transforms
from fdq.misc import print_nb_weights
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots, get_norm_to_rgb

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

    norm_to_rgb = get_norm_to_rgb(experiment)

    data = experiment.data["celeb_HDF"]
    model = experiment.models["monaivae"]
    targs = experiment.exp_def.train.args

    train_loader = data.train_data_loader

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        model.train()
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)
            
            if experiment.useAMP:

                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )

                with torch.autocast(device_type=device_type, enabled=True):
                    reconstruction, z_mu, z_sigma = model(images_gt)
                    train_loss_tensor = (
                        experiment.losses["MAE"](images_gt, reconstruction)
                        / experiment.gradacc_iter
                    )
                    
                experiment.scaler.scale(train_loss_tensor).backward()
            
            else:
                reconstruction, z_mu, z_sigma = model(images_gt)
                train_loss_tensor = (
                        experiment.losses["MAE"](images_gt, reconstruction)
                        / experiment.gradacc_iter
                    )
                
                train_loss_tensor.backward()
            
            experiment.update_gradients(
                b_idx=nb_tbatch, loader_name="celeb_HDF", model_name="monaivae"
            )
            
        pbar.finish()
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation train_loss_tensor
        
        epsilon = torch.randn_like(z_sigma)
        z_sample = z_mu + z_sigma * epsilon
        
        
        imgs_to_log =[
                {"name": "input", "data": images_gt},
                {"name": "recon", "data": reconstruction},
                {"name": "sample", "data": z_sample},
                {"name": "mu", "data": z_mu},
                {"name": "sigma", "data": z_sigma},
            ]
        
        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break