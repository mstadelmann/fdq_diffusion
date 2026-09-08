import torch
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb
from chuchichaestli.diffusion.ddpm import DDPM
from image_functions import createSubplots


@torch.no_grad()
def get_sample_from_noise(experiment, diffuser, gen_shape, idx_to_store=None, condition=None):
    targs = experiment.cfg.train.args
    model = experiment.get_eval_model(targs.model_name)
    model.eval()

    imgs = []

    for j, img in enumerate(
        diffuser.generate(
            model=model,
            shape=tuple(gen_shape),
            condition=condition,
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
    model = experiment.models[targs.model_name]

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
    is_grayscale = False

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
                condition = batch[1].to(experiment.device) if len(batch) == 2 else None
            else:
                images_gt = batch.to(experiment.device)
                condition = None

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
                    imgs_to_log.append({"name": "train_cond", "path": cond_imgs_path})

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(
                    x_t=images_gt, condition=condition
                )
                noise_pred = model(noisy_imgs, t)
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
                    condition = batch[1].to(experiment.device) if len(batch) == 2 else None
                else:
                    images_gt = batch.to(experiment.device)
                    condition = None

                n_val_processed += images_gt.shape[0]
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(
                    x_t=images_gt, condition=condition
                )
                noise_pred = model(noisy_imgs, t)
                val_loss_sum += experiment.losses["MSE"](noise, noise_pred).item()

            experiment.valLoss = val_loss_sum / n_val_processed

            # Additional dummy validation: generate samples
            idx_to_store = torch.linspace(
                0,
                targs.diffusion_nb_steps - 1,
                targs.get("diffusion_nb_plot_steps", 15),
                dtype=torch.int,
            ).tolist()

            # These preview plots pass explicit per-row/per-step labels sized for a single
            # sample (e.g. one "GT"/"Condition"/"Generated" row, one image per diffusion
            # step) - createSubplots silently explodes a multi-sample batch into one image
            # per sample, which would then no longer match those labels. Generate from a
            # single validation sample here so the shapes the labels expect are the shapes
            # actually produced (mirrors the equivalent fix already in diffusion_monai.py).
            sample_condition = condition[:1] if condition is not None else None
            images = get_sample_from_noise(
                experiment=experiment,
                diffuser=chuchi_diffuser,
                gen_shape=img_shape,
                idx_to_store=idx_to_store,
                condition=sample_condition,
            )

            history_path = createSubplots(
                image_list=images,
                grayscale=is_grayscale,
                experiment=experiment,
                histogram=True,
                figure_title="Generative Diffusion Steps",
                labels=[f"Step {i}" for i in idx_to_store],
                export_transform=t_img_exp,
            )

            imgs_to_log.extend(
                [
                    {"name": "gen_result", "data": t_img_exp(images[-1])},
                    {"name": "gen_hist", "path": history_path},
                ]
            )

            if condition is not None:
                cond_diff_path = createSubplots(
                    image_list=[
                        images_gt[:1],
                        sample_condition,
                        images[-1],
                        images_gt[:1] - images[-1],
                    ],
                    grayscale=is_grayscale,
                    experiment=experiment,
                    histogram=True,
                    figure_title="Generative Diffusion Steps",
                    export_transform=t_img_exp,
                    labels=["GT", "Condition", "Generated", "GT - Generated"],
                    apply_global_range=False,
                )
                imgs_to_log.append({"name": "gen_cond_diff", "path": cond_diff_path})

            experiment.on_epoch_end(log_images_wandb=imgs_to_log)

            if experiment.check_early_stop():
                break


@torch.no_grad()
def fdq_test(experiment):
    # This test is only for generative diffusion!

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
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

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)
    test_iter = iter(data.test_data_loader)

    chuchi_diffuser = DDPM(
        num_timesteps=targs.diffusion_nb_steps,
        device=experiment.device,
        beta_start=targs.diffusion_shd_beta_start,
        beta_end=targs.diffusion_shd_beta_end,
        schedule=targs.diffusion_scheduler,
    )

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
        else:
            images_gt = test_sample.to(experiment.device)
            condition = None

        if img_shape is None:
            img_shape = images_gt.shape[1:]
            is_grayscale = img_shape[0] == 1

        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            images = get_sample_from_noise(
                experiment=experiment,
                diffuser=chuchi_diffuser,
                gen_shape=img_shape,
                idx_to_store=idx_to_store,
                condition=condition,
            )

        history_path = createSubplots(
            image_list=images,
            grayscale=is_grayscale,
            experiment=experiment,
            histogram=True,
            figure_title="Generative Diffusion Steps",
            labels=[f"Step {i}" for i in idx_to_store],
            export_transform=t_img_exp,
        )

        save_wandb(
            experiment, images={"name": "test_gen_history", "path": history_path}
        )

        gts.append(images_gt.cpu())
        if condition is not None:
            conditions.append(condition.cpu())
        results.append(images[-1].cpu())

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

    save_wandb(experiment, images={"name": "test_samples", "path": res_path})
