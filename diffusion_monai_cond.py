import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint
from monai.inferers import DiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from image_functions import createSubplots

# https://github.com/Project-MONAI/tutorials/blob/main/generation/2d_ddpm/2d_ddpm_tutorial.ipynb


@torch.no_grad()
def fdq_test(experiment):
    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

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

    results = []

    nb_test_samples = experiment.exp_def.test.args.get("nb_test_samples", 10)
    pbar = startProgBar(nb_test_samples, "testing...")
    model.eval()

    for nb_tbatch, batch in enumerate(data.test_data_loader):
        pbar.update(nb_tbatch)

        if nb_tbatch >= nb_test_samples:
            break

        with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
            images_gt = batch[0].to(experiment.device)
            condition = batch[1].to(experiment.device)
            noise = torch.randn((1, *images_gt.shape[1:]), device=experiment.device)
            scheduler.set_timesteps(num_inference_steps=targs.diffusion_nb_steps)

            isteps = int(
                targs.diffusion_nb_steps / targs.get("diffusion_nb_plot_steps", 15)
            )

            is_grayscale = noise.shape[1] == 1

            image, intermediates = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=isteps,
                conditioning=condition,
                mode="concat",
            )

        results.append(image.cpu())

        if norm_to_hu is not None:
            result_hu = norm_to_hu(image)
            images_gt_hu = norm_to_hu(images_gt)
            PSNR = experiment.losses["PSNR"](images_gt_hu, result_hu).item()
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

        img_list = [
            images_gt,
            condition,
            image,
            images_gt - image,
            images_gt_hu,
            result_hu,
        ]
        img_labels = [
            "GT",
            "Condition",
            "Generated",
            "GT - Generated",
            "GT_HU",
            "Generated_HU",
        ]

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

        imgs_to_log = [
            {"name": "diffusion steps", "path": history_path},
            {"name": "results", "path": cond_diff_path},
        ]

        save_wandb(experiment, images=imgs_to_log, scalars=log_scalars)

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

    save_wandb(experiment, images={"name": "results overview", "path": res_path})

    pbar.finish()
    print("Test samples generated and saved.")
