"""Flow matching training/sampling on top of fdq.

Uses the flow_matching library (CC-BY-NC 4.0, installed as a dependency, not part of
this repository): https://github.com/facebookresearch/flow_matching
    Lipman et al., "Flow Matching Guide and Code", arXiv:2412.06264, 2024.
    Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023, arXiv:2210.02747.
The overall training/sampling procedure (conditional OT probability path, velocity
regression, midpoint ODE solver) follows the library's image example
(examples/image/training/train_loop.py, eval_loop.py), re-implemented here for fdq.
"""

import gc
import math

import torch
from torchvision.utils import make_grid
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb

from flow_matching.path import CondOTProbPath
from flow_matching.utils import ModelWrapper
from flow_matching.solver.ode_solver import ODESolver


def wandb_image_entry(name: str, images: torch.Tensor, as_grid: bool) -> dict:
    """Build a save_wandb/on_epoch_end image entry.

    If `as_grid` is True, the batch is merged into a single grid image (one glance
    overview). Otherwise each image is logged individually, as before (click-through
    gallery in the wandb UI).
    """
    if as_grid:
        nrow = max(1, math.isqrt(images.shape[0]))
        images = make_grid(images, nrow=nrow, padding=2)
    return {"name": name, "data": images}


class GuidedVelocityModel(ModelWrapper):
    """Velocity field wrapper for the ODE solver with classifier-free guidance.

    v = (1 + w) * v(x, t | label) - w * v(x, t), with guidance weight w = cfg_scale
    (Ho & Salimans, "Classifier-Free Diffusion Guidance", arXiv:2207.12598, 2022).
    For an unconditional model (num_classes: null) the label is ignored by the UNet,
    so both terms are identical and the result equals the plain model output.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor):
        # the solver passes a scalar time, the UNet expects one value per sample
        t_batch = torch.zeros(x.shape[0], device=x.device) + t
        with torch.no_grad(), torch.amp.autocast("cuda"):
            v_cond = self.model(x, t_batch, extra={"label": label})
            if cfg_scale == 0.0:
                v = v_cond
            else:
                v_uncond = self.model(x, t_batch, extra={})
                v = v_cond + cfg_scale * (v_cond - v_uncond)
        return v.float()


class WeightEMA:
    """Exponential moving average of a model's trainable parameters.

    The decay is warmed up as min(decay, (1 + n) / (10 + n)) over the number of
    updates n, so early (still poor) weights are forgotten quickly. The averaged
    weights are not saved separately: use swap_in() to temporarily load them into
    the model (e.g. for sampling/checkpointing) and swap_out() to go back to the
    raw training weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.n_updates = 0
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.averaged = [p.detach().clone() for p in self.params]
        self.stash: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self) -> None:
        self.n_updates += 1
        d = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        for avg, p in zip(self.averaged, self.params):
            avg.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self) -> None:
        self.stash = [p.detach().clone() for p in self.params]
        for p, avg in zip(self.params, self.averaged):
            p.copy_(avg)

    @torch.no_grad()
    def swap_out(self) -> None:
        for p, raw in zip(self.params, self.stash):
            p.copy_(raw)
        self.stash = []


@torch.no_grad()
def fdq_test(experiment):
    targs = experiment.cfg.train.args
    model = experiment.models_no_ddp[targs.model_name]
    data = experiment.data[targs.dataloader_name]

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]
    log_as_grid = experiment.cfg.store.get("wandb_img_as_grid", True)

    # infer the resolution to generate at from the actual (post-transform) test
    # images - a hardcoded value here silently mismatches configs that train/test
    # at a resolution other than 128 (e.g. the x256 experiments), producing
    # incoherent samples since the model never trained at that resolution.
    test_sample = next(iter(data.test_data_loader))
    if isinstance(test_sample, (list, tuple)):
        test_sample = test_sample[0]
    sample_resolution = test_sample.shape[-1]

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)
    pbar = startProgBar(nb_test_samples, "evaluation...")
    results = []

    for inf_nb in range(nb_test_samples):
        pbar.update(inf_nb)

        batch_size = 16

        labels = torch.tensor(
            list(range(batch_size)), dtype=torch.int32, device=experiment.device
        )

        cfg_weighted_model = GuidedVelocityModel(model=model)

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

        synthetic_samples = t_img_exp(synthetic_samples)

        save_wandb(
            experiment=experiment,
            images=[wandb_image_entry("eval_samples", synthetic_samples, log_as_grid)],
        )


def _epoch_mean_loss(experiment, loss_sum: float, sample_count: int) -> float:
    """Reduce a rank-local (loss_sum, sample_count) pair to the true epoch-wide mean loss.

    Each rank only sums the loss over its own DistributedSampler shard, so under DDP
    the local sum/count must be all-reduced across ranks before dividing - otherwise
    the logged loss depends on world_size instead of just reflecting the model's fit.
    """
    if experiment.is_distributed():
        stats = torch.tensor(
            [loss_sum, sample_count], dtype=torch.float64, device=experiment.device
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        loss_sum, sample_count = stats[0].item(), stats[1].item()
    return loss_sum / sample_count


def fdq_train(experiment) -> None:
    iprint("Meta FlowMatching Training")

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]
    log_as_grid = experiment.cfg.store.get("wandb_img_as_grid", True)

    targs = experiment.cfg.train.args
    data = experiment.data[targs.dataloader_name]
    model = experiment.models[targs.model_name]
    model_no_ddp = experiment.models_no_ddp[targs.model_name]

    ema_decay = targs.get("ema_decay")
    ema = WeightEMA(model_no_ddp, decay=ema_decay) if ema_decay else None

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"
    train_loader = data.train_data_loader

    path = CondOTProbPath()

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.on_epoch_start(epoch=epoch)

        model.train(True)
        train_loss_sum = 0.0
        train_sample_count = 0
        pbar = startProgBar(data.n_train_batches, "training...")
        gc.collect()

        # for data_iter_step, (samples, labels) in enumerate(data_loader):
        for data_iter_step, batch in enumerate(train_loader):
            pbar.update(data_iter_step)

            samples = batch[0] if isinstance(batch, (list, tuple)) else batch

            samples = samples.to(experiment.device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            # if torch.rand(1) < args.class_drop_prob:
            # if torch.rand(1) < 0.2:
            #     conditioning = {}
            # else:
            #     conditioning = {"label": labels}

            conditioning = {}
            noise = torch.randn_like(samples).to(experiment.device)

            t = torch.rand(samples.shape[0]).to(experiment.device)

            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                train_loss_tensor = torch.pow(
                    model(x_t, t, extra=conditioning) - u_t, 2
                ).mean()

            batch_size = samples.shape[0]
            train_loss_sum += train_loss_tensor.detach().item() * batch_size
            train_sample_count += batch_size

            train_loss_tensor = train_loss_tensor / experiment.gradacc_iter

            if experiment.useAMP:
                experiment.scaler.scale(train_loss_tensor).backward()
            else:
                train_loss_tensor.backward()

            apply_update = ((data_iter_step + 1) % experiment.gradacc_iter == 0) or (
                data_iter_step + 1 == data.n_train_batches
            )
            experiment.update_gradients(
                b_idx=data_iter_step,
                loader_name=targs.dataloader_name,
                model_name=targs.model_name,
            )

            if apply_update and ema is not None:
                ema.update()

        pbar.finish()
        experiment.trainLoss = _epoch_mean_loss(experiment, train_loss_sum, train_sample_count)

        model.eval()
        val_loader = data.val_data_loader
        val_loss_sum = 0.0
        val_sample_count = 0
        pbar = startProgBar(data.n_val_batches, "validation...")

        with torch.no_grad():
            for val_iter_step, batch in enumerate(val_loader):
                pbar.update(val_iter_step)

                samples = batch[0] if isinstance(batch, (list, tuple)) else batch
                samples = samples.to(experiment.device, non_blocking=True)

                conditioning = {}
                noise = torch.randn_like(samples).to(experiment.device)

                t = torch.rand(samples.shape[0]).to(experiment.device)

                path_sample = path.sample(t=t, x_0=noise, x_1=samples)
                x_t = path_sample.x_t
                u_t = path_sample.dx_t

                with torch.autocast(device_type=device_type, enabled=experiment.useAMP):
                    val_loss_tensor = experiment.losses["MSE"](
                        model(x_t, t, extra=conditioning), u_t
                    )

                batch_size = samples.shape[0]
                val_loss_sum += val_loss_tensor.item() * batch_size
                val_sample_count += batch_size

        pbar.finish()
        experiment.valLoss = _epoch_mean_loss(experiment, val_loss_sum, val_sample_count)

        # swap in EMA weights for sample generation + checkpointing, so the logged
        # preview and the saved/resumable model reflect the smoothed weights rather
        # than this epoch's raw weights
        if ema is not None:
            ema.swap_in()

        # generate samples for wandb logging - resolution follows the actual
        # (post-transform) training images, not a hardcoded value, so previews stay
        # consistent with whatever resolution this config actually trains at
        batch_size = 16
        sample_resolution = samples.shape[-1]

        labels = torch.tensor(
            list(range(batch_size)), dtype=torch.int32, device=experiment.device
        )

        # use the un-wrapped module for sampling: repeatedly running forward-only
        # inference through the DDP-wrapped model (no matching backward) can
        # desync DDP's reducer bookkeeping across ranks and eventually hang the
        # next real training-step all-reduce.
        cfg_weighted_model = GuidedVelocityModel(model=model_no_ddp)

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

        synthetic_samples = t_img_exp(synthetic_samples)

        experiment.on_epoch_end(
            log_images_wandb=[wandb_image_entry("samples", synthetic_samples, log_as_grid)]
        )

        if ema is not None:
            ema.swap_out()
