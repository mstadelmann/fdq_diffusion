import torch


class PSNR(torch.nn.Module):
    """
    Pytorch module to calculate PSNR

    https://github.com/fpschill/ac3t/blob/master/ac3t/metrics.py

    :param img: image1 tensor (ground truth)
    :param img2: image2 tensor (predicted)
    :param mode: "norm": for data-range [-1, 1]
                 "HU": for datarange [-1000, 1000]
                 "PSNR_manual": set range manually
    :return: scalar output PSNR(img, img2)

    max is the maximum fluctuation in the input image.
    -> double-precision float: max = 1
    -> 8-bit unsigned int: max = 255

    according to https://pytorch.org/ignite/generated/ignite.metrics.PSNR.html
    The data range of the target image (distance between minimum and maximum possible values).

    i.e. for HU, we consider images in [-1000, 1000], so max = 2000
    -> we could add an offset of 1000 to the images, which would result
    images [0, 2000] without influencing the MSE.
    """

    def __init__(self, mode="norm"):
        super(PSNR, self).__init__()
        self.mode = mode
        self.warning_printed = False  # print warning messages only once
        if mode not in ["PSNR_manual", "norm", "HU"]:
            raise ValueError(
                f"Invalid mode for PSNR, must be 'PSNR_manual', 'norm' or 'HU', but was {mode}"
            )

    def forward(self, img, img2, top_man=None):

        if self.mode == "HU":
            top = 2 * 1000
        elif self.mode == "PSNR_manual":
            if top_man is None:
                raise ValueError(
                    "'top_man' must be manually defined when using PSNR_manual!"
                )
            top = float(top_man)
        else:
            # self.mode == "norm"
            top = 2 * 1

        mse = torch.mean((img - img2) ** 2)
        return 10 * torch.log10(top**2 / mse)
