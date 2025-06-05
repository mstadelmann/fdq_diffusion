import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def compute_histo(image, nb_bins=100):
    bins = torch.linspace(image.min(), image.max(), nb_bins).cpu()
    return torch.histogram(image.cpu().float(), bins=bins)


def createSubplots(
    image_list,
    batch_nb=0,
    labels=None,
    grayscale=False,
    experiment=None,
    histogram=False,
    save_path=None,
    figure_title=None,
    max_cols=None,
    hide_ticks=False,
    apply_global_range=True,
    fig_size=3,
    export_transform=None,
    show_colorbar=True,
):

    def _ax(axis, r=None, c=None):
        # Helper function to access axes in a flexible way
        ndim = np.ndim(axis)
        if ndim == 0:
            return axis
        if ndim == 1:
            return axis[r]
        if ndim == 2:
            return axis[r, c]

    if labels is None:
        labels = len(image_list) * [""]
    elif len(labels) != len(image_list):
        raise ValueError("ERROR - nb labels does not correspond to nb images!")

    nb_images = len(image_list)

    if export_transform is None:
        export_transform = transforms.Lambda(lambda t: t)

    glob_print_min = min(float(export_transform(img).min()) for img in image_list)
    glob_print_max = max(float(export_transform(img).max()) for img in image_list)

    if histogram:
        nb_rows = 2
        nb_cols = nb_images
    elif nb_images < 3:
        nb_rows = 1
        nb_cols = nb_images
    else:
        nb_rows = 2
        nb_cols = int(np.ceil(nb_images / nb_rows))

    if not histogram and max_cols is not None and max_cols < nb_cols:
        nb_cols = max_cols
        nb_rows = int(np.ceil(nb_images / nb_cols))

    fig, axs = plt.subplots(
        nb_rows, nb_cols, figsize=(fig_size * nb_cols, fig_size * nb_rows)
    )

    if figure_title is not None:
        plt.suptitle(figure_title)

    if not hide_ticks:
        plt.subplots_adjust(wspace=0.4)

    for i, (img, lbl) in enumerate(zip(image_list, labels)):
        is_last = i == len(image_list) - 1

        # to make this work for numpy arrays
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # subplot(nrows, ncols..)
        c_row = int(np.floor(i / nb_cols))
        c_col = i - (c_row * nb_cols)

        if len(img.shape) > 3:
            img = img[batch_nb, ...]
            img = torch.squeeze(img, 0)

        if len(img.shape) == 3:
            img = img.permute(1, 2, 0)

        if histogram:
            if grayscale:
                hist = compute_histo(torch.squeeze(img).cpu())
                _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="r")
                # fix_histo_ticks(axs[1, c_col], hist)

            else:
                hist = compute_histo(torch.squeeze(img).cpu()[0])
                _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="r")
                hist = compute_histo(torch.squeeze(img).cpu()[1])
                _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="g")
                hist = compute_histo(torch.squeeze(img).cpu()[2])
                _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="b")

        img = export_transform(img)

        if not apply_global_range:
            glob_print_min = float(img.min())
            glob_print_max = float(img.max())

        cmap = "gray" if grayscale else None
        im = _ax(axs, c_row, c_col).imshow(
            img.cpu(), cmap=cmap, vmin=glob_print_min, vmax=glob_print_max
        )
        if show_colorbar:
            if not apply_global_range:
                fig.colorbar(im, ax=_ax(axs, c_row, c_col))

            elif is_last:
                fig.colorbar(im, ax=_ax(axs, c_row, c_col))

        if lbl is not None:
            _ax(axs, c_row, c_col).set_title(lbl)

    if hide_ticks:
        if np.ndim(axs) == 0:
            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_xticklabels([])
            axs.set_yticklabels([])
        else:
            for ax in axs.flat:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    if save_path is None:
        save_path = experiment.get_next_export_fn()

    plt.savefig(save_path)

    return save_path
