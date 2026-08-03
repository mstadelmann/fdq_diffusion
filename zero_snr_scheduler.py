import torch
from monai.networks.schedulers import DDPMScheduler


class ZeroSNRDDPMScheduler(DDPMScheduler):
    """DDPMScheduler with betas rescaled so alphas_cumprod reaches exactly 0 at the last timestep.

    Standard linear beta schedules (e.g. beta_start=1e-4, beta_end=0.02) leave a small but nonzero
    signal-to-noise ratio at t=T-1. During training the "noisiest" sample therefore still carries a
    faint trace of the clean image, dominated by its low-frequency content (overall color/brightness).
    At inference, sampling starts from true zero-mean Gaussian noise, which lacks that trace, so the
    model has to guess the global color from nothing and produces inconsistent color casts per sample.
    Rescaling betas to force alphas_cumprod[-1] == 0 removes this train/inference mismatch.
    See Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed" (2023).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.betas = self._rescale_zero_terminal_snr(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        sqrt_0 = alphas_bar_sqrt[0].clone()
        sqrt_T = alphas_bar_sqrt[-1].clone()

        alphas_bar_sqrt = alphas_bar_sqrt - sqrt_T
        alphas_bar_sqrt = alphas_bar_sqrt * sqrt_0 / (sqrt_0 - sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[:1], alphas])

        return 1.0 - alphas
