import torch
from torchvision import transforms
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint
from monai.inferers import DiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from image_functions import createSubplots

# https://github.com/Project-MONAI/tutorials/blob/main/generation/2d_ddpm/2d_ddpm_tutorial.ipynb


def fdq_train(experiment) -> None:
    iprint("MONAI Diffusion Training")

    img_exp_op = experiment.exp_def.store.img_exp_transform
    if img_exp_op is None:
        t_img_exp = transforms.Lambda(lambda t: t)
    else:
        t_img_exp = experiment.transformers[img_exp_op]

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    train_loader = data.train_data_loader

    # MONAI DDPMScheduler
    # Args:
    #     num_train_timesteps: number of diffusion steps used to train the model.
    #     schedule: member of NoiseSchedules, name of noise schedule function in component store
    #     variance_type: member of DDPMVarianceType
    #     clip_sample: option to clip predicted sample between -1 and 1 for numerical stability.
    #     prediction_type: member of DDPMPredictionType
    #     clip_sample_min: minimum clipping value when clip_sample equals True
    #     clip_sample_max: maximum clipping value when clip_sample equals True
    #     schedule_args: arguments to pass to the schedule function

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

    img_shape = None
    is_grayscale = False

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
                if img_shape[0] == 1:
                    is_grayscale = True

                # first test batch: store inputs
                gt_imgs_path = createSubplots(
                    image_list=images_gt,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    hide_ticks=True,
                    figure_title="Input GT",
                    export_transform=t_img_exp,
                )
                imgs_to_log.append({"name": "train_imgs", "path": gt_imgs_path})

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):

                noise = torch.randn_like(images_gt).to(experiment.device)
                timesteps = torch.randint(
                    0,
                    targs.diffusion_nb_steps,
                    (images_gt.shape[0],),
                    device=images_gt.device,
                ).long()

                noise_pred = inferer(
                    inputs=images_gt,
                    diffusion_model=model,
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

        model.eval()
        pbar = startProgBar(data.n_val_batches, "validation...")
        val_loss_sum = 0.0

        with torch.no_grad(), torch.autocast(
            device_type=device_type, enabled=experiment.useAMP
        ):

            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch)

                images_gt = batch[0].to(experiment.device)
                noise = torch.randn_like(images_gt).to(experiment.device)
                timesteps = torch.randint(
                    0,
                    targs.diffusion_nb_steps,
                    (images_gt.shape[0],),
                    device=images_gt.device,
                ).long()

                noise_pred = inferer(
                    inputs=images_gt,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                )

                val_loss_sum += experiment.losses["MSE"](noise, noise_pred).item()

            experiment.valLoss = val_loss_sum / len(data.val_data_loader.dataset)
            pbar.finish()

            # Generate sample
            noise = torch.randn((1, *images_gt.shape[1:]), device=experiment.device)
            scheduler.set_timesteps(num_inference_steps=targs.diffusion_nb_steps)

            isteps = int(
                targs.diffusion_nb_steps / targs.get("diffusion_nb_plot_steps", 15)
            )

            image, intermediates = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=isteps,
            )

        history_path = createSubplots(
            image_list=intermediates,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps",
            export_transform=t_img_exp,
        )

        imgs_to_log.extend(
            [
                {"name": "gen_result", "data": t_img_exp(image)},
                {"name": "gen_hist", "path": history_path},
            ]
        )

        experiment.finalize_epoch(log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break


@torch.no_grad()
def fdq_test(experiment):

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]
    test_loader = data.test_data_loader

    img_exp_op = experiment.exp_def.store.img_exp_transform
    if img_exp_op is None:
        t_img_exp = transforms.Lambda(lambda t: t)
    else:
        t_img_exp = experiment.transformers[img_exp_op]

    if experiment.exp_def.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
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

    nb_test_samples = experiment.exp_def.test.args.get("nb_test_samples", 10)
    test_sample = next(iter(test_loader))[0]

    pbar = startProgBar(nb_test_samples, "evaluation...")

    results = []

    for inf_nb in range(nb_test_samples):
        pbar.update(inf_nb)

        # Generate sample
        noise = torch.randn((1, *test_sample.shape[1:]), device=experiment.device)
        scheduler.set_timesteps(num_inference_steps=targs.diffusion_nb_steps)

        is_grayscale = noise.shape[1] == 1

        isteps = int(
            targs.diffusion_nb_steps / targs.get("diffusion_nb_plot_steps", 15)
        )

        image, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=isteps,
        )

        _ = createSubplots(
            image_list=intermediates,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title=f"Test img {inf_nb + 1}",
            export_transform=t_img_exp,
            show_colorbar=False,
        )

        results.append(image.cpu())

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
    imgs_to_log = [
        {"name": "test_samples", "path": res_path},
    ]

    save_wandb(experiment, images=imgs_to_log)

    pbar.finish()
    print("Test samples generated and saved.")
