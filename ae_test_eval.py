"""Shared test evaluation for the CBCT autoencoder experiments (VAE, MAISI, Wan2.2),
so that all of them log the same metrics and plots to wandb:

- per test sample: PSNR_test / PSNR_test_HU and a "test_sample" figure
  (GT / recon / diff [/ HU] with axial slice, histogram, sagittal and coronal slice)
- at the end: a "results_overview" figure (axial slice of every reconstruction)
  and the mean of every per-sample scalar as "test_mean/<name>".
"""

from collections import defaultdict

import torch
from fdq.misc import save_wandb
from image_functions import create_sample_views_figure, createSubplots

# (row label, transformer name) of the additional views; skipped if not defined in the config
EXTRA_VIEWS = [
    ("sagittal", "extract_2d_from_3d_sag"),
    ("coronal", "extract_2d_from_3d_cor"),
]


class AETestEvaluator:
    def __init__(self, experiment, model_name):
        self.experiment = experiment
        self.model_name = model_name

        exp_trans_name = experiment.cfg.store.img_exp_transform
        if exp_trans_name is None:
            raise ValueError(
                "Experiment definition must contain an 'img_exp_transform' entry!"
            )
        self.t_axial = experiment.transformers[exp_trans_name]
        self.views = [("axial", self.t_axial)] + [
            (label, experiment.transformers[name])
            for label, name in EXTRA_VIEWS
            if name in experiment.transformers
        ]
        self.norm_to_hu = experiment.transformers.get("norm_to_HU")

        self.scores = defaultdict(list)
        self.recon_axial = []

    def add_sample(
        self, nb_sample, images_gt, recon, extra_images=None, extra_scalars=None
    ):
        """Evaluate and log one test sample (batch size 1)."""
        images_gt = images_gt.float()
        recon = recon.float()

        scalars = {"PSNR_test": self.experiment.losses["PSNR"](images_gt, recon).item()}
        categories = [
            ("GT", images_gt),
            ("Recon", recon),
            ("Diff (GT - Recon)", images_gt - recon),
        ]

        if self.norm_to_hu is not None:
            gt_hu = self.norm_to_hu(images_gt)
            recon_hu = self.norm_to_hu(recon)
            scalars["PSNR_test_HU"] = self.experiment.losses["PSNR_hu"](
                gt_hu, recon_hu
            ).item()
            categories.extend([("GT HU", gt_hu), ("Recon HU", recon_hu)])

        if extra_scalars is not None:
            scalars.update(extra_scalars)

        for k, v in scalars.items():
            self.scores[k].append(v)

        self.recon_axial.append(self.t_axial(recon.cpu()))

        title = f"{self.model_name} - test sample {nb_sample}: " + ", ".join(
            f"{k} {v:.2f}" for k, v in scalars.items() if k.startswith("PSNR")
        )
        fig_path = create_sample_views_figure(
            categories=categories,
            views=self.views,
            experiment=self.experiment,
            figure_title=title,
        )

        images = [{"name": "test_sample", "path": fig_path}]
        if extra_images is not None:
            images.extend(extra_images)

        save_wandb(self.experiment, images=images, scalars=scalars)

    def finish(self):
        """Log the reconstruction overview and the mean scores, return the means."""
        means = {k: float(torch.tensor(v).mean()) for k, v in self.scores.items()}
        print(means)

        overview_path = createSubplots(
            image_list=self.recon_axial,
            labels=[f"sample {i}" for i in range(len(self.recon_axial))],
            grayscale=True,
            experiment=self.experiment,
            figure_title=f"{self.model_name} - reconstructions (axial)",
            max_cols=10,
            hide_ticks=True,
            apply_global_range=False,
            show_colorbar=False,
        )

        save_wandb(
            self.experiment,
            images=[{"name": "results_overview", "path": overview_path}],
            scalars={f"test_mean/{k}": v for k, v in means.items()},
        )

        return means
