import torch
import sys

from fdq.ui_functions import startProgBar, iprint
from fdq.misc import print_nb_weights


def nb_requested_train_inferences(experiment):
    # get optional diffusion inference schedule, to speed up training
    # expected format: {"ep_0": {"freq": 5,"nb_inf": 2},...}

    dis = experiment.expfile.get("store", {}).get("diffusion_inference_schedule", None)

    if experiment.current_epoch == 0 or dis is None:
        nb_inferences = 1
    else:
        try:
            dis_key_epochs = sorted([int(k.split("ep_")[1]) for k in dis.keys()])
            current_rule = dis_key_epochs[0]
            for key_ep in dis_key_epochs:
                if key_ep <= experiment.current_epoch:
                    current_rule = key_ep

            current_freq = dis[f"ep_{current_rule}"]["freq"]
            nb_inferences = dis[f"ep_{current_rule}"]["nb_inf"]

            if (experiment.current_epoch % current_freq) != 0:
                nb_inferences = 0

        except Exception as e:
            print(f"ERROR - Unable to parse diffusion_inference_schedule: {e}")
            nb_inferences = 1

    return nb_inferences


def linear_unscaled_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(
        beta_start, beta_end, timesteps
    )  # [beta_start ... beta_end] in T steps


def linear_scaled_beta_schedule(timesteps, beta_start, beta_end):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start_s = scale * beta_start
    beta_end_s = scale * beta_end
    return torch.linspace(beta_start_s, beta_end_s, timesteps)


def cosine_beta_schedule(timesteps, s):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])


def sigmoid_unscaled_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def sigmoid_scaled_beta_schedule(timesteps, start, end, tau):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    # t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps TODO why was this float64? necessary?
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])


class DiffusionSchedule:
    """
    Refs:
    https://learnopencv.com/denoising-diffusion-probabilistic-models/
    """

    def __init__(self, experiment) -> None:
        device = experiment.device
        timesteps = experiment.train_params.get("diffusion_nb_steps")
        self.timesteps = timesteps
        self.device = device

        beta_schedule = experiment.train_params.get("diffusion_scheduler")

        if beta_schedule == "linear_unscaled":
            beta_start = experiment.train_params.get("diffusion_shd_beta_start")
            beta_end = experiment.train_params.get("diffusion_shd_beta_end")
            if timesteps is None or beta_start is None or beta_end is None:
                print("ERROR - diffusion parameters not defined!")
                sys.exit()

            self.betas = linear_unscaled_beta_schedule(timesteps, beta_start, beta_end)

        elif beta_schedule == "linear_scaled":
            beta_start = experiment.train_params.get("diffusion_shd_beta_start")
            beta_end = experiment.train_params.get("diffusion_shd_beta_end")
            if timesteps is None or beta_start is None or beta_end is None:
                print("ERROR - diffusion parameters not defined!")
                sys.exit()

            self.betas = linear_scaled_beta_schedule(timesteps, beta_start, beta_end)

        elif beta_schedule == "cosine":
            s = experiment.train_params.get("diffusion_shd_cos_s")
            if s is None:
                print("ERROR - diffusion parameter 'diffusion_shd_cos_s' not defined!")
                sys.exit()
            self.betas = cosine_beta_schedule(timesteps, s=s)

        elif beta_schedule == "sigmoid_unscaled":
            beta_start = experiment.train_params.get("diffusion_shd_beta_start")
            beta_end = experiment.train_params.get("diffusion_shd_beta_end")
            self.betas = sigmoid_unscaled_beta_schedule(
                timesteps, beta_start=beta_start, beta_end=beta_end
            )

        elif beta_schedule == "sigmoid_scaled":
            start = experiment.train_params.get("diffusion_shd_sigmoid_start")
            end = experiment.train_params.get("diffusion_shd_sigmoid_end")
            tau = experiment.train_params.get("diffusion_shd_sigmoid_tau")
            self.betas = sigmoid_scaled_beta_schedule(
                timesteps, start=start, end=end, tau=tau
            )

        else:
            print(f"ERROR - Unknown beta schedule '{beta_schedule}'!")
            sys.exit()

        clip_min = experiment.train_params.get("diffusion_shd_clip_min")
        clip_max = experiment.train_params.get("diffusion_shd_clip_max")

        if clip_min is None:
            clip_min = self.betas.min()
        if clip_max is None:
            clip_max = self.betas.max()

        self.betas = torch.clip(self.betas, clip_min, clip_max).to(device)

        self.alphas = 1.0 - self.betas  # min = 0.98, max = 0.9999

        self.alphas_hat = torch.cumprod(
            self.alphas, dim=0
        ).to(
            device  # [max = 0.9999 ... min = 0.0481] --> min gets very small with large T (9e-8 for T=1600)
        )
        self._sq_alphas_hat = torch.sqrt(self.alphas_hat).to(
            device
        )  # [max = 0.9999 ... min = 0.2192]
        self._sq_one_minus_alphas_hat = torch.sqrt(1.0 - self.alphas_hat).to(
            device
        )  # [min = 0.01 ... max = 0.9757]
        self.sq_one_div_alphas = torch.sqrt(1.0 / self.alphas).to(
            device
        )  # [min = 1.0 ... max = 1.0102]

        self.alphas_hat_prev = F.pad(
            self.alphas_hat[:-1], (1, 0), value=1.0
        )  # [max = 1.0 ... min = 0.049]
        self.sigma_square = (
            self.betas
            * (1.0 - self.alphas_hat_prev)
            / (1.0 - self.alphas_hat)  # [min = 0 ... max = 0.01998]
        )

    def sq_alphas_hat(self, indices, shape):
        batch_size = indices.shape[0]
        out = self._sq_alphas_hat.gather(-1, indices)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(self.device)

    def sq_one_minus_alphas_hat(self, indices, shape):
        batch_size = indices.shape[0]
        out = self._sq_one_minus_alphas_hat.gather(-1, indices)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(self.device)


def forward_diffusion(image, t, diffusion_schedule):
    """
    Returns noisy version of image
    x_0 = original image
    x_T = Pure Noise


    one step
    ---------
    q(x_t|x_t-1) := N{x_t; sqrt(1-B_t)x_t-1, B_t*I}
                                |             |-> variance
                                | -> centered (mean)
                                     if B = 1 -> output centred at 0
    Given large T, x_T is a nearly isotropic Gaussian distribution

    B = Diffusion rate -> precalculated according to schedule
    In the original DDPM paper, B = linear = [0.0001, 0.02]


    xt = sqrt(1-B) * x_t-1 + sqrt(B_t) * epsilon
    epsilon = N(0, I)
    xt = sqrt(1-B) * x_t-1 + sqrt(B_t) * N(0, I)


    closed form
    -----------

    alpha = 1 - B -> alpha_t = 1 - B_t
    alpha_hat = cum_prod(alpha_t) -> alpha_t_hat = cum_prod(alpha_t)
    cum_prod(alpha_t) = alpha_1 * alpha_2 * ... * alpha_t [i=1..t]

    q(x_t|x_0) = N{x_t; sqrt(alpha_t_hat)x_0, sqrt(1 - alpha_t_hat) * I}

    x_t = sqrt(alpha_t_hat) * x_0 + sqrt(1 - alpha_t_hat) * epsilon
    epsilon = N(0, I)
    """
    # Sample new noise from normal distribution
    # noise.shape = [B, C, H, W]
    noise = torch.randn(image.size(), device=image.device)

    # MEAN
    # sq_alpha_t_hat.shape = [B,1,1,1]
    sq_alpha_t_hat = diffusion_schedule.sq_alphas_hat(indices=t, shape=image.shape)
    # mean.shape = [B, C, H, W]
    mean = sq_alpha_t_hat * image

    # VARIANCE
    # sq_one_minus_alpha_t_hat.shape = [B,1,1,1]
    sq_one_minus_alpha_t_hat = diffusion_schedule.sq_one_minus_alphas_hat(
        indices=t, shape=image.shape
    )
    # variance.shape = [B, C, H, W]
    variance = sq_one_minus_alpha_t_hat * noise

    return mean + variance, noise


def get_loss(
    experiment, images, steps, diffusion_schedule, static_concat_condition=None
):
    x_noisy, noise = forward_diffusion(images, steps, diffusion_schedule)
    if static_concat_condition is not None:
        x_noisy = torch.cat((static_concat_condition, x_noisy), dim=1)
    noise_pred = experiment.networkModel(x_noisy, steps)
    return experiment.lossFunction(noise, noise_pred)


@torch.no_grad()
def sample_ddpm_initial_condition(
    experiment, diffusion_schedule, start_img, nb_plot_steps, nb_denoise_steps
):
    img = start_img.to(experiment.device)

    img_history = []

    if nb_plot_steps == 0:
        plot_indices = []
    else:
        plot_indices = torch.linspace(
            0, nb_denoise_steps, nb_plot_steps, dtype=torch.int32
        )

    pbar = startProgBar(nb_denoise_steps, "Sampling...")

    # 2) for t in T, starting with the highest on going to 1
    for i in range(0, nb_denoise_steps)[::-1]:
        pbar.update(nb_denoise_steps - i)
        t = torch.tensor([i], device=experiment.device)

        # 3) z ~ N(0, I) if t > 0 else z = 0
        if i > 0:
            # normal distribution around zero
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        betas_t = diffusion_schedule.betas[t].reshape((1, 1, 1, 1))
        sq_one_div_alphas_t = diffusion_schedule.sq_one_div_alphas[t].reshape(
            (1, 1, 1, 1)
        )
        sq_one_minus_alphas_hat_t = diffusion_schedule._sq_one_minus_alphas_hat[
            t
        ].reshape((1, 1, 1, 1))

        predicted_noise = experiment.networkModel(img, t)

        # 4) see formula above
        img = (
            sq_one_div_alphas_t
            * (img - (betas_t / sq_one_minus_alphas_hat_t) * predicted_noise)
            + torch.sqrt(diffusion_schedule.sigma_square[t].reshape((1, 1, 1, 1)))
            * noise
        )

        if i in plot_indices:
            img_history.append(img)

        img = experiment.samplingTransformer(img)

    # if we dont store specific images, only return the last one
    if len(plot_indices) == 0:
        img_history.append(img)

    pbar.finish()

    return img_history


def train(experiment) -> None:
    iprint("Vanilla Diffusion Training")
    print_nb_weights(experiment)

    # max 15 plot steps
    nb_plot_steps = min(
        experiment.expfile.get("store", {}).get("nb_plot_steps", 10), 15
    )

    diffusion_schedule = DiffusionSchedule(experiment)

    train_loader = experiment.dataPreparator.train_data_loader

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        experiment.networkModel.train()

        training_loss_value = 0.0

        n_train = experiment.dataPreparator.n_train
        pbar = startProgBar(n_train, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch * experiment.train_batch_size)
            images_gt = batch[0].to(experiment.device)

            if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
                # forward-pass first train sample to check if noise schedule makes sense
                # run_forward_pass(
                #     experiment,
                #     diffusion_schedule,
                #     img_input=images_gt,
                #     nb_plots=10,
                #     custom_step_nb=diffusion_schedule.timesteps,
                # )
                # plot_diffusion_schedule(experiment, diffusion_schedule)

                if experiment.net_input_size is not None:
                    raise ValueError(
                        "ERROR - net_input_size was manually defined in experiment file."
                    )
                # ignore batch size
                experiment.net_input_size = [1] + list(images_gt.shape[1:])

            nb_samples = len(images_gt)
            t = torch.randint(
                0, diffusion_schedule.timesteps, (nb_samples,), device=experiment.device
            )

            # this can be written without code repetition, however, goal is to keep both options flexible...
            # following: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            if experiment.useAMP:
                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )
                # torch.autocast(dtype=torch.float16) does not work on CPU
                # torch.autocast(dtype=torch.bfloat16) does not work on V100
                # torch.autocast(NO DTYPE) on V100 uses the same amount of mem as torch.float16

                with torch.autocast(device_type=device_type, enabled=True):
                    loss = (
                        get_loss(
                            experiment=experiment,
                            images=images_gt,
                            steps=t,
                            diffusion_schedule=diffusion_schedule,
                        )
                        / experiment.gradacc_iter
                    )

                # Scale the loss and back-propagate
                experiment.scaler.scale(loss).backward()

            else:
                loss = (
                    get_loss(
                        experiment=experiment,
                        images=images_gt,
                        steps=t,
                        diffusion_schedule=diffusion_schedule,
                    )
                    / experiment.gradacc_iter
                )
                loss.backward()

            experiment.update_gradients(b_idx=nb_tbatch)

            training_loss_value += loss.data.item() * images_gt.size(
                0
            )  # TODO shape[0] ?

        pbar.finish()

        # img = sample_ddpm(
        #     experiment, diffusion_schedule, nb_plot_steps, debug_img_export=True
        # )

        experiment.trainLoss = training_loss_value / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation dataset

        iprint(f"Training Loss: {experiment.trainLoss:.4f}")

        # pylint: disable=W0631
        experiment.compute_memory_requirements(train_batch=batch)

        # finalize_epoch(experiment)
        experiment.finalize_epoch()

        if experiment.check_early_stop():
            break
