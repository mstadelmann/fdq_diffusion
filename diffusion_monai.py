import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint
from monai.inferers import DiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from image_functions import createSubplots, stack_images_vertically
from zero_snr_scheduler import ZeroSNRDDPMScheduler


def _build_scheduler(targs, scheduler_args):
    prediction_type = targs.get("diffusion_prediction_type", "epsilon")
    zero_terminal_snr = targs.get("diffusion_zero_terminal_snr", False)

    if zero_terminal_snr and prediction_type != "v_prediction":
        raise ValueError(
            "diffusion_zero_terminal_snr=True requires diffusion_prediction_type='v_prediction'. "
            "With epsilon prediction, alpha_prod_t hits exactly 0 at the last training timestep, "
            "which makes the predicted-x0 division blow up to +/-inf on the first sampling step."
        )

    scheduler_cls = ZeroSNRDDPMScheduler if zero_terminal_snr else DDPMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=targs.diffusion_nb_steps,
        schedule=targs.diffusion_scheduler,
        prediction_type=prediction_type,
        **scheduler_args,
    )
    return scheduler, prediction_type


# https://github.com/Project-MONAI/tutorials/blob/main/generation/2d_ddpm/2d_ddpm_tutorial.ipynb


def fdq_train(experiment) -> None:
    iprint("MONAI Diffusion Training")

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    if "norm_to_HU" in experiment.transformers:
        norm_to_hu = experiment.transformers["norm_to_HU"]
    else:
        norm_to_hu = None

    targs = experiment.cfg.train.args
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
        for param in scheduler_args.keys():
            if param not in supported_scheduler_args:
                raise ValueError(
                    f"Unsupported scheduler argument: {param}. "
                    f"Supported arguments are: {supported_scheduler_args}"
                )

    scheduler, prediction_type = _build_scheduler(targs, scheduler_args)
    inferer = DiffusionInferer(scheduler)

    img_shape = None
    is_grayscale = False

    # Optional auxiliary perceptual loss on top of the standard noise/velocity-prediction
    # MSE. Off by default (lpips_weight=0.0) - existing experiments that don't define
    # 'lpips_weight' or a 'LPIPS' loss are completely unaffected.
    lpips_weight = targs.get("lpips_weight", 0.0)
    lpips_loss_fn = experiment.losses["LPIPS"].to(experiment.device).eval() if lpips_weight > 0 else None

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)
        imgs_to_log = []

        model.train()
        train_loss_sum = 0.0
        n_train_processed = 0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)

            if isinstance(batch, (list, tuple)):
                images_gt = batch[0].to(experiment.device)
                if len(batch) == 2:
                    condition = batch[1].to(experiment.device)
                    monai_mode = "concat"
                else:
                    condition = None
                    monai_mode = "crossattn"  # default MONAI mode
            else:
                images_gt = batch.to(experiment.device)
                condition = None
                monai_mode = "crossattn"  # default MONAI mode

            n_train_processed += images_gt.shape[0]

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # run dummy forward pass to visualize noise schedule
                # TODO

                # Plot diffusion schedule
                # TODO

                # store input img
                img_shape = images_gt.shape[1:]
                if img_shape[0] == 1:
                    is_grayscale = True

                # first test batch: store inputs. GT and (if present) Condition are
                # built as separate 2-row (images + histogram) figures, then stacked
                # into one 4-row image so they're easier to compare side by side
                # instead of as two separate wandb panels.
                gt_imgs_path = createSubplots(
                    image_list=images_gt,
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    hide_ticks=True,
                    figure_title="Input GT",
                    export_transform=t_img_exp,
                )

                if condition is not None:
                    cond_imgs_path = createSubplots(
                        image_list=condition,
                        grayscale=is_grayscale,
                        experiment=experiment,
                        histogram=True,
                        hide_ticks=True,
                        figure_title="Input Condition",
                        export_transform=t_img_exp,
                    )
                    combined_path = stack_images_vertically(
                        [gt_imgs_path, cond_imgs_path],
                        save_path=experiment.get_next_export_fn(name="train_gt_cond"),
                    )
                    imgs_to_log.append({"name": "train_imgs", "path": combined_path})
                else:
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
                    condition=condition,
                    mode=monai_mode,
                )

                target = (
                    noise
                    if prediction_type == "epsilon"
                    else scheduler.get_velocity(images_gt, noise, timesteps)
                )
                loss_tensor = experiment.losses["MSE"](target, noise_pred)

                if lpips_loss_fn is not None:
                    # Reconstruct the estimated clean image x0_hat from the model output
                    # (the same closed-form the scheduler's own step() uses), then
                    # compare it to GT with LPIPS - the noise/velocity-space MSE above
                    # under-penalizes localized texture/detail loss.
                    noisy_images = scheduler.add_noise(
                        original_samples=images_gt, noise=noise, timesteps=timesteps
                    )
                    t_shape = [-1] + [1] * (images_gt.dim() - 1)
                    alpha_prod_t = scheduler.alphas_cumprod.to(device=images_gt.device)[
                        timesteps
                    ].reshape(t_shape)
                    beta_prod_t = 1 - alpha_prod_t
                    if prediction_type == "epsilon":
                        x0_hat = (noisy_images - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
                    else:  # v_prediction
                        x0_hat = alpha_prod_t.sqrt() * noisy_images - beta_prod_t.sqrt() * noise_pred
                    # x0_hat is a poor/noisy estimate at high t (early in the forward
                    # process) - clamp to the data range rather than trying to weight it
                    # away, keeping this simple.
                    x0_hat = x0_hat.clamp(-1, 1)
                    loss_tensor = loss_tensor + lpips_weight * lpips_loss_fn(x0_hat, images_gt).mean()

                train_loss_tensor = loss_tensor / experiment.gradacc_iter
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
        experiment.trainLoss = train_loss_sum / n_train_processed

        model.eval()
        pbar = startProgBar(data.n_val_batches, "validation...")
        val_loss_sum = 0.0
        n_val_processed = 0

        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, enabled=experiment.useAMP),
        ):
            for nb_vbatch, batch in enumerate(data.val_data_loader):
                pbar.update(nb_vbatch)

                if isinstance(batch, (list, tuple)):
                    images_gt = batch[0].to(experiment.device)
                    if len(batch) == 2:
                        condition = batch[1].to(experiment.device)
                    else:
                        condition = None
                else:
                    images_gt = batch.to(experiment.device)
                    condition = None

                n_val_processed += images_gt.shape[0]
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
                    condition=condition,
                    mode=monai_mode,
                )

                target = (
                    noise
                    if prediction_type == "epsilon"
                    else scheduler.get_velocity(images_gt, noise, timesteps)
                )
                val_loss_sum += experiment.losses["MSE"](target, noise_pred).item()

            experiment.valLoss = val_loss_sum / n_val_processed
            pbar.finish()

            # Generate sample
            noise = torch.randn((1, *images_gt.shape[1:]), device=experiment.device)
            scheduler.set_timesteps(num_inference_steps=targs.diffusion_nb_steps)

            isteps = int(
                targs.diffusion_nb_steps / targs.get("diffusion_nb_plot_steps", 15)
            )

            sample_condition = condition[:1] if condition is not None else None
            image, intermediates = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=isteps,
                conditioning=sample_condition,
                mode=monai_mode,
            )

            if norm_to_hu is not None and condition is not None:
                result_hu = norm_to_hu(image)
                images_gt_hu = norm_to_hu(images_gt[:1])
                PSNR = experiment.losses["PSNR"](images_gt[:1], image).item()
                PSNR_hu = experiment.losses["PSNR_hu"](images_gt_hu, result_hu).item()
                log_scalars = {"PSNR": PSNR, "PSNR_hu": PSNR_hu}
            else:
                log_scalars = None

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

        if condition is not None:
            # images_gt/condition are still the full validation batch; image (and
            # sample_condition it was generated from) is a single sample - slice down to
            # match, otherwise createSubplots explodes the batch into one image per
            # sample and the fixed 4/6-entry labels below no longer match the image count.
            img_list = [images_gt[:1], sample_condition, image, images_gt[:1] - image]
            img_labels = ["GT", "Condition", "Generated", "GT - Generated"]
            if norm_to_hu is not None:
                img_list.extend([images_gt_hu, result_hu])
                img_labels.extend(["GT_HU", "Generated_HU"])

            cond_diff_path = createSubplots(
                image_list=img_list,
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="Generative Diffusion Steps",
                export_transform=t_img_exp,
                labels=img_labels,
                apply_global_range=False,
            )

            imgs_to_log.extend(
                [
                    {
                        "name": "gen_cond_diff",
                        "path": cond_diff_path,
                    }
                ]
            )

        experiment.on_epoch_end(log_scalars=log_scalars, log_images_wandb=imgs_to_log)
        if experiment.check_early_stop():
            break


@torch.no_grad()
def fdq_test(experiment):
    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]
    test_loader = data.test_data_loader

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    if experiment.cfg.data.get(targs.dataloader_name).args.test_batch_size != 1:
        raise ValueError(
            "Error: Test batch size must be 1 for this experiment. Please change the experiment file."
        )

    supported_scheduler_args = ["beta_start", "beta_end", "sig_range", "s"]

    scheduler_args = targs.get("diffusion_scheduler_args", {})
    if scheduler_args is None:
        scheduler_args = {}
    else:
        for param in scheduler_args.keys():
            if param not in supported_scheduler_args:
                raise ValueError(
                    f"Unsupported scheduler argument: {param}. "
                    f"Supported arguments are: {supported_scheduler_args}"
                )

    scheduler, _ = _build_scheduler(targs, scheduler_args)
    inferer = DiffusionInferer(scheduler)

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)
    test_iter = iter(test_loader)

    pbar = startProgBar(nb_test_samples, "evaluation...")

    gts, conditions, results = [], [], []
    img_shape = None
    is_grayscale = None

    for inf_nb in range(nb_test_samples):
        pbar.update(inf_nb)

        # fetch a fresh sample/condition pair each iteration - reusing the
        # same one for every sample made every "different" test result
        # condition on an identical source image.
        test_sample = next(test_iter)
        if isinstance(test_sample, (list, tuple)):
            images_gt = test_sample[0].to(experiment.device)
            condition = (
                test_sample[1].to(experiment.device)
                if len(test_sample) == 2
                else None
            )
            monai_mode = "concat" if condition is not None else "crossattn"
        else:
            images_gt = test_sample.to(experiment.device)
            condition = None
            monai_mode = "crossattn"

        if img_shape is None:
            img_shape = images_gt.shape[1:]
            is_grayscale = img_shape[0] == 1

        # Generate sample
        noise = torch.randn((1, *img_shape), device=experiment.device)
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
            conditioning=condition,
            mode=monai_mode,
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

        gts.append(images_gt.cpu())
        if condition is not None:
            conditions.append(condition.cpu())
        results.append(image.cpu())

    pbar.finish()

    if conditions:
        # 4 rows: ground truth / condition / result / ground truth - result
        gt_batch = torch.cat(gts, dim=0)
        result_batch = torch.cat(results, dim=0)
        res_path = createSubplots(
            image_list=[
                gt_batch,
                torch.cat(conditions, dim=0),
                result_batch,
                gt_batch - result_batch,
            ],
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=False,
            figure_title="Generated test samples (rows: ground truth / condition / result / GT - result)",
            export_transform=t_img_exp,
            hide_ticks=True,
            show_colorbar=False,
            apply_global_range=False,
            max_cols=nb_test_samples,
        )
    else:
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

    print("Test samples generated and saved.")
