import torch
from torchvision import transforms
from fdq.misc import print_nb_weights
from fdq.ui_functions import startProgBar, iprint
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots, get_norm_to_rgb


@torch.no_grad()
def get_sample_from_noise(experiment, diffuser, gen_shape, idx_to_store=None):
    model = experiment.models["ccUNET"]
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
    print_nb_weights(experiment)

    norm_to_rgb = get_norm_to_rgb(experiment)

    data = experiment.data["celeb_HDF"]
    model = experiment.models["ccUNET"]
    targs = experiment.exp_def.train.args

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

        model.train()
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                img_shape = images_gt.shape[1:]

                # first test batch: store inputs
                gt_imgs_path = createSubplots(
                    image_list=[norm_to_rgb(img) for img in images_gt],
                    grayscale=False,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Input GT",
                )
                imgs_to_log.append({"name": "train_imgs", "path": gt_imgs_path})

            # this can be written without code repetition, however, goal is to keep both options flexible...
            # following: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            if experiment.useAMP:
                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )

                with torch.autocast(device_type=device_type, enabled=True):
                    noisy_imgs, noise, t = chuchi_diffuser.noise_step(images_gt)
                    noise_pred = model(noisy_imgs, t)
                    train_loss_tensor = (
                        experiment.losses["MAE"](noise, noise_pred)
                        / experiment.gradacc_iter
                    )

                # noise_pred becomes NaN if AMP is used. Why?
                # https://pytorch.org/docs/stable/amp.html
                # https://pytorch.org/docs/2.2/notes/amp_examples.html#amp-examples

                experiment.scaler.scale(train_loss_tensor).backward()

            else:
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(images_gt)
                noise_pred = model(noisy_imgs, t)
                train_loss_tensor = (
                    experiment.losses["MAE"](noise, noise_pred)
                    / experiment.gradacc_iter
                )

                train_loss_tensor.backward()

            experiment.update_gradients(
                b_idx=nb_tbatch, loader_name="celeb_HDF", model_name="ccUNET"
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

        images = get_sample_from_noise(
            experiment=experiment,
            diffuser=chuchi_diffuser,
            gen_shape=img_shape,
            idx_to_store=idx_to_store,
        )

        images = [norm_to_rgb(i) for i in images]

        history_path = createSubplots(
            image_list=images,
            grayscale=False,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps",
            labels=[f"Step {i}" for i in idx_to_store],
        )

        imgs_to_log.extend(
            [
                {"name": "gen_result", "data": images[-1]},
                {"name": "gen_hist", "path": history_path},
            ]
        )

        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break
