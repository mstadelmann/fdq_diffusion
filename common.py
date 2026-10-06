import torch


class PSNR(torch.nn.Module):
    """Peak signal-to-noise ratio, PSNR = 10 * log10(data_range^2 / MSE), in dB.

    The data range is the distance between the smallest and largest possible
    value of the images, set by `mode`:
        "norm":        images normalized to [-1, 1]      -> data_range = 2
        "HU":          images in Hounsfield units, window
                       [-1000, 1000]                     -> data_range = 2000
        "PSNR_manual": data_range passed per call as `top_man`

    Usage: PSNR(mode)(img, img2), where img is the reference (ground truth).
    """

    DATA_RANGES = {"norm": 2.0, "HU": 2000.0}

    def __init__(self, mode="norm"):
        super().__init__()
        if mode not in ("norm", "HU", "PSNR_manual"):
            raise ValueError(
                f"Invalid mode for PSNR, must be 'PSNR_manual', 'norm' or 'HU', but was {mode}"
            )
        self.mode = mode

    def forward(self, img, img2, top_man=None):
        if self.mode == "PSNR_manual":
            if top_man is None:
                raise ValueError("'top_man' (data range) is required in PSNR_manual mode!")
            data_range = float(top_man)
        else:
            data_range = self.DATA_RANGES[self.mode]

        mse = torch.mean((img - img2) ** 2)
        return 10 * torch.log10(data_range**2 / mse)
