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
):
    if labels is None:
        labels = len(image_list) * [""]
    elif len(labels) != len(image_list):
        raise ValueError("ERROR - nb labels does not correspond to nb images!")

    nb_images = len(image_list)

    global_min = min(float(img.min()) for img in image_list)
    global_max = max(float(img.max()) for img in image_list)

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

        # if not grayscale and img.dtype == torch.float and img.max() > 50:
        #     # assume RGB image in float values
        #     # TODO where is this used? this might not be a good idea!
        #     img = img.int()

        if not apply_global_range:
            global_min = float(img.min())
            global_max = float(img.max())

        if grayscale:
            if nb_rows > 1:
                im = axs[c_row, c_col].imshow(
                    img.cpu(), cmap="gray", vmin=global_min, vmax=global_max
                )
                if not apply_global_range:
                    fig.colorbar(im, ax=axs[c_row, c_col])

                elif is_last:
                    fig.colorbar(im, ax=axs[c_row, c_col])
            else:
                axs[c_col].imshow(
                    img.cpu(), cmap="gray", vmin=global_min, vmax=global_max
                )
        else:
            if nb_rows > 1:
                axs[c_row, c_col].imshow(img.cpu(), vmin=global_min, vmax=global_max)
            else:
                axs[c_col].imshow(img.cpu(), vmin=global_min, vmax=global_max)

        if lbl is not None:
            if nb_rows > 1:
                axs[c_row, c_col].set_title(lbl)
            else:
                axs[c_col].set_title(lbl)

        if histogram:
            if grayscale:
                hist = compute_histo(torch.squeeze(img).cpu())
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="r")
                # fix_histo_ticks(axs[1, c_col], hist)

            else:
                hist = compute_histo(torch.squeeze(img).cpu()[0])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="r")
                hist = compute_histo(torch.squeeze(img).cpu()[1])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="g")
                hist = compute_histo(torch.squeeze(img).cpu()[2])
                axs[1, c_col].plot(hist.bin_edges[:-1], hist.hist, color="b")

    if hide_ticks:
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if save_path is None:
        save_path = experiment.get_next_export_fn()

    plt.savefig(save_path)

    return save_path
