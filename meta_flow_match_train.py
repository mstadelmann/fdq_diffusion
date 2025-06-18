import torch
from fdq.misc import save_wandb
from fdq.ui_functions import startProgBar, iprint

import math
from flow_matching.path import CondOTProbPath


from meta_ema import EMA
from meta_grad_scaler import NativeScalerWithGradNormCount

from typing import Iterable
import gc

from meta_flow_match_eval import CFGScaledModel
from flow_matching.solver.ode_solver import ODESolver

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def fdq_train(experiment) -> None:
    iprint("Meta FlowMatching Training")

    exp_trans_name = experiment.exp_def.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]

    targs = experiment.exp_def.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]
    # lr_schedule = experiment.lr_schedulers[targs.model_name]
    optimizer = experiment.optimizers[targs.model_name]
    loss_scaler = NativeScalerWithGradNormCount()

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"

    train_loader = data.train_data_loader

    skewed_timesteps = False

    path = CondOTProbPath()

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")
        imgs_to_log = []

        model.train(True)
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")
        gc.collect()

        # for data_iter_step, (samples, labels) in enumerate(data_loader):
        for data_iter_step, batch in enumerate(train_loader):
            pbar.update(data_iter_step)

            samples = batch[0]

            if data_iter_step % experiment.gradacc_iter == 0:
                optimizer.zero_grad()  # this is done in update_gradients

            samples = samples.to(experiment.device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            # if torch.rand(1) < args.class_drop_prob:
            # if torch.rand(1) < 0.2:
            #     conditioning = {}
            # else:
            #     conditioning = {"label": labels}

            conditioning = {}
            noise = torch.randn_like(samples).to(experiment.device)

            if skewed_timesteps:
                t = skewed_timestep_sample(samples.shape[0], device=experiment.device)
            else:
                t = torch.torch.rand(samples.shape[0]).to(experiment.device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                train_loss_tensor = torch.pow(
                    model(x_t, t, extra=conditioning) - u_t, 2
                ).mean()

            train_loss_tensor /= experiment.gradacc_iter

            apply_update = (data_iter_step + 1) % experiment.gradacc_iter == 0
            loss_scaler(
                train_loss_tensor,
                optimizer,
                parameters=model.parameters(),
                update_grad=apply_update,
            )
            # if apply_update and isinstance(model, EMA):
            #     model.update_ema()
            # elif (
            #     apply_update
            #     and isinstance(model, DistributedDataParallel)
            #     and isinstance(model.module, EMA)
            # ):
            #     model.module.update_ema()
            train_loss_sum += train_loss_tensor.detach().item()

        pbar.finish()
        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)

        # lr_schedule.step()  # this is done in finalize EP

        # dummy validation
        experiment.valLoss = 0
        batch_size = 16
        sample_resolution = 128

        labels = torch.tensor(
            list(range(batch_size)), dtype=torch.int32, device=experiment.device
        )

        cfg_weighted_model = CFGScaledModel(model=model)

        x_0 = torch.randn(
            [batch_size, 3, sample_resolution, sample_resolution],
            dtype=torch.float32,
            device=experiment.device,
        )
        solver = ODESolver(velocity_model=cfg_weighted_model)

        synthetic_samples = solver.sample(
            time_grid=torch.tensor([0.0, 1.0], device=experiment.device),
            x_init=x_0,
            method="midpoint",
            atol=None,
            rtol=None,
            step_size=0.01,
            label=labels,
            cfg_scale=0.2,
        )

        # Scaling to [0, 1] from [-1, 1]
        # synthetic_samples = torch.clamp(synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0)
        # synthetic_samples = torch.floor(synthetic_samples * 255) / 255.0

        synthetic_samples = t_img_exp(synthetic_samples)

        experiment.finalize_epoch(
            log_images_wandb=[{"name": "samples", "data": synthetic_samples}]
        )
