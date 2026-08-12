import gc
import logging
import math

import torch
from torch.nn import Module, Parameter, ParameterList
from torchvision.utils import make_grid
from fdq.ui_functions import startProgBar, iprint
from fdq.misc import save_wandb

from flow_matching.path import CondOTProbPath
from flow_matching.utils import ModelWrapper
from flow_matching.solver.ode_solver import ODESolver

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


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


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor):
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.amp.autocast("cuda"), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.amp.autocast("cuda"), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


# Exponential moving average of model weights. Wired into fdq_train: shadow params are
# updated after every optimizer step, then swapped into the model (via backup/copy_to_model
# /restore_to_model) for sample generation and checkpointing, so training itself always
# resumes from the raw (non-averaged) weights.
class EMA(Module):
    def __init__(self, model: Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params: ParameterList = ParameterList(
            [
                Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params: list[torch.Tensor] = []

    def train(self, mode: bool) -> None:
        if self.training == mode:
            super().train(mode)
            return

        if not mode:
            logger.info(
                "EMA: Switching from train to eval, backing up parameters and copying EMA params"
            )
            self.backup()
            self.copy_to_model()
        else:
            logger.info("EMA: Switching from eval to train, restoring saved parameters")
            self.restore_to_model()

        super().train(mode)

    def update_ema(self) -> None:
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        assert (
            self.training
        ), "Backup can only be created in train mode to avoid backing-up ema weights."
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)


@torch.no_grad()
def fdq_test(experiment):
    targs = experiment.cfg.train.args
    model = experiment.models[targs.model_name]

    exp_trans_name = experiment.cfg.store.img_exp_transform
    if exp_trans_name is None:
        raise ValueError(
            "Experiment definition must contain an 'img_exp_transform' entry!"
        )
    t_img_exp = experiment.transformers[exp_trans_name]
    log_as_grid = experiment.cfg.store.get("wandb_img_as_grid", True)

    nb_test_samples = experiment.cfg.test.args.get("nb_test_samples", 10)
    pbar = startProgBar(nb_test_samples, "evaluation...")
    results = []

    for inf_nb in range(nb_test_samples):
        pbar.update(inf_nb)

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
    ema = EMA(model_no_ddp, decay=ema_decay) if ema_decay else None

    device_type = "cuda" if experiment.device == torch.device("cuda") else "cpu"
    train_loader = data.train_data_loader
    skewed_timesteps = False

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
                ema.update_ema()

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

                if skewed_timesteps:
                    t = skewed_timestep_sample(samples.shape[0], device=experiment.device)
                else:
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
            ema.backup()
            ema.copy_to_model()

        # generate samples for wandb logging
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

        synthetic_samples = t_img_exp(synthetic_samples)

        experiment.on_epoch_end(
            log_images_wandb=[wandb_image_entry("samples", synthetic_samples, log_as_grid)]
        )

        if ema is not None:
            ema.restore_to_model()
