import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def stack_images_vertically(paths, save_path):
    """Vertically stack multiple already-saved images into one combined image.

    Useful to combine separate createSubplots() outputs (e.g. one 2-row GT+histogram
    figure and one 2-row Condition+histogram figure) into a single figure, rather than
    logging them as separate images.

    Args:
        paths: List of image file paths to stack, top to bottom. None entries (e.g. a
            createSubplots() call that failed and returned None) are skipped.
        save_path: Path to write the combined image to.

    Returns:
        str: save_path, or None if fewer than one valid path was given.
    """
    imgs = [Image.open(p) for p in paths if p is not None]
    if not imgs:
        return None
    if len(imgs) == 1:
        imgs[0].save(save_path)
        return save_path

    width = max(im.width for im in imgs)
    total_height = sum(im.height for im in imgs)
    combined = Image.new(imgs[0].mode, (width, total_height), color="white")

    y = 0
    for im in imgs:
        combined.paste(im, (0, y))
        y += im.height

    combined.save(save_path)
    return save_path


def compute_histo(image, nb_bins=100):
    bins = torch.linspace(image.min(), image.max(), nb_bins).cpu()
    return torch.histogram(image.cpu().float(), bins=bins)


@torch.no_grad()
def create_category_grid(
    categories,
    grayscale=False,
    experiment=None,
    save_path=None,
    figure_title=None,
    fig_size=3,
    export_transform=None,
    show_colorbar=False,
    hide_ticks=True,
):
    """Build a grid figure with one column per sample and, for each category
    (e.g. ground truth / reconstruction / difference), one image row directly
    followed by one histogram row.

    Args:
        categories: list of (label, tensors) tuples. `tensors` is a list of
            per-sample tensors/arrays (one column each); every category must
            provide the same number of samples.
        grayscale (bool): whether images are single-channel.
        experiment (class, optional): experiment object, used to derive
            `save_path` and to skip plotting on non-main processes.
        save_path (str, optional): where to write the figure. Derived from
            `experiment` if not given.
        figure_title (str, optional): overall figure title.
        fig_size (float): size (inches) of a single grid cell.
        export_transform (callable, optional): transform applied to each
            sample tensor before plotting (e.g. denormalization).
        show_colorbar (bool): whether to draw a colorbar next to each image.
        hide_ticks (bool): whether to hide axis ticks on the image rows.

    Returns:
        str: save_path the figure was written to, or None on failure / if not
        the main process.
    """
    fig = None
    try:
        if save_path is None and experiment is not None:
            if hasattr(experiment, "is_main_process") and not experiment.is_main_process():
                return None

        if export_transform is None:
            export_transform = transforms.Lambda(lambda t: t)

        nb_cols = len(categories[0][1])
        nb_rows = 2 * len(categories)

        fig, axs = plt.subplots(
            nb_rows, nb_cols, figsize=(fig_size * nb_cols, fig_size * nb_rows)
        )
        axs = np.atleast_2d(axs)
        if axs.shape != (nb_rows, nb_cols):
            axs = axs.reshape(nb_rows, nb_cols)

        if figure_title is not None:
            plt.suptitle(figure_title)

        for cat_idx, (label, tensors) in enumerate(categories):
            if len(tensors) != nb_cols:
                raise ValueError(
                    f"ERROR - category '{label}' has {len(tensors)} samples, expected {nb_cols}!"
                )

            img_row = 2 * cat_idx
            hist_row = img_row + 1

            for col, img in enumerate(tensors):
                img = (
                    img.detach().cpu()
                    if isinstance(img, torch.Tensor)
                    else torch.tensor(img)
                )
                img_t = export_transform(img).cpu()

                if len(img_t.shape) > 3:
                    img_t = img_t[0, ...]
                    img_t = torch.squeeze(img_t, 0)

                if grayscale:
                    hist = compute_histo(img_t)
                    axs[hist_row, col].plot(hist.bin_edges[:-1], hist.hist, color="r")
                else:
                    sq = torch.squeeze(img_t)
                    for ch, color in zip(range(min(3, sq.shape[0])), "rgb"):
                        hist = compute_histo(sq[ch])
                        axs[hist_row, col].plot(hist.bin_edges[:-1], hist.hist, color=color)

                plot_t = img_t
                if not grayscale and len(plot_t.shape) == 3:
                    plot_t = plot_t.permute(1, 2, 0)

                cmap = "gray" if grayscale else None
                im = axs[img_row, col].imshow(
                    plot_t,
                    cmap=cmap,
                    vmin=float(img_t.min()),
                    vmax=float(img_t.max()),
                )
                if show_colorbar:
                    fig.colorbar(im, ax=axs[img_row, col])

                if col == 0:
                    axs[img_row, col].set_ylabel(label)
                    axs[hist_row, col].set_ylabel(f"hist {label}")

                if cat_idx == 0:
                    axs[img_row, col].set_title(f"sample {col}")

                axs[hist_row, col].set_yticks([])
                axs[hist_row, col].set_yticklabels([])

                if hide_ticks:
                    axs[img_row, col].set_xticks([])
                    axs[img_row, col].set_yticks([])
                    axs[img_row, col].set_xticklabels([])
                    axs[img_row, col].set_yticklabels([])

        if save_path is None:
            if experiment is None:
                raise ValueError("create_category_grid requires save_path or experiment.")
            save_path = experiment.get_next_export_fn()

        if save_path is None:
            return None

        plt.savefig(save_path)

        return save_path

    except Exception as e:
        print(f"An error occurred while creating category grid: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)


@torch.no_grad()
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
            if c is None:
                return axis[r]
            elif r is None:
                return axis[c]
            else:
                if c > 0:
                    return axis[c]
                else:
                    return axis[r]
        if ndim == 2:
            return axis[r, c]

    fig = None
    try:
        if save_path is None and experiment is not None:
            if hasattr(experiment, "is_main_process") and not experiment.is_main_process():
                return None

        if not isinstance(image_list, list):
            image_list = [image_list]

        # to make this work for numpy arrays
        image_list_tensor = [
            (
                torch.tensor(i).detach().cpu()
                if not isinstance(i, torch.Tensor)
                else i.detach().cpu()
            )
            for i in image_list
        ]

        # if batchsize > 1, flatten the batch dimension
        # this assumes that all images in the list have the same shape!
        if image_list_tensor[0].ndim > 3 and image_list_tensor[0].shape[0] > 1:
            image_list_tensor = [
                x.unsqueeze(0) for imgt in image_list_tensor for x in imgt
            ]

        if labels is None:
            labels = len(image_list_tensor) * [""]
        elif len(labels) != len(image_list_tensor):
            raise ValueError("ERROR - nb labels does not correspond to nb images!")

        nb_images = len(image_list_tensor)

        if export_transform is None:
            export_transform = transforms.Lambda(lambda t: t)

        image_list_trans = [export_transform(i).cpu() for i in image_list_tensor]

        if len(image_list_trans[0].shape) > 4:
            raise ValueError(
                "ERROR - images with more than 4 dimensions are not supported! Add an appropriate transformation!"
            )

        glob_print_min = min(float(i.min()) for i in image_list_trans)
        glob_print_max = max(float(i.max()) for i in image_list_trans)

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

        for i, (img, img_t, lbl) in enumerate(
            zip(image_list_tensor, image_list_trans, labels)
        ):
            is_last = i == len(image_list_tensor) - 1

            # subplot(nrows, ncols..)
            c_row = int(np.floor(i / nb_cols))
            c_col = i - (c_row * nb_cols)

            if histogram:
                if grayscale:
                    hist = compute_histo(img)
                    _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="r")
                    # fix_histo_ticks(axs[1, c_col], hist)

                else:
                    hist = compute_histo(torch.squeeze(img)[0])
                    _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="r")
                    hist = compute_histo(torch.squeeze(img)[1])
                    _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="g")
                    hist = compute_histo(torch.squeeze(img)[2])
                    _ax(axs, 1, c_col).plot(hist.bin_edges[:-1], hist.hist, color="b")

            if not apply_global_range:
                glob_print_min = float(img_t.min())
                glob_print_max = float(img_t.max())

            if len(img_t.shape) > 3:
                img_t = img_t[batch_nb, ...]
                img_t = torch.squeeze(img_t, 0)

            if not grayscale and len(img_t.shape) == 3:
                img_t = img_t.permute(1, 2, 0)

            cmap = "gray" if grayscale else None
            im = _ax(axs, c_row, c_col).imshow(
                img_t, cmap=cmap, vmin=glob_print_min, vmax=glob_print_max
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
                if histogram:
                    if np.ndim(axs) == 1:
                        # if we have only 1 img with histogram, its still 1D
                        axs[0].set_xticks([])
                        axs[0].set_yticks([])
                        axs[0].set_xticklabels([])
                        axs[0].set_yticklabels([])
                    else:
                        for i in range(nb_cols):
                            axs[0, i].set_xticks([])
                            axs[0, i].set_yticks([])
                            axs[0, i].set_xticklabels([])
                            axs[0, i].set_yticklabels([])
                else:
                    for ax in axs.flat:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
        else:
            if histogram:
                for i in range(nb_cols):
                    if np.ndim(axs) == 1:
                        axs[i].set_yticks([])
                        axs[i].set_yticklabels([])
                    else:
                        axs[1, i].set_yticks([])
                        axs[1, i].set_yticklabels([])

        if save_path is None:
            if experiment is None:
                raise ValueError("createSubplots requires save_path or experiment.")
            save_path = experiment.get_next_export_fn()

        if save_path is None:
            return None

        plt.savefig(save_path)

        return save_path

    except Exception as e:
        print(f"An error occurred while creating subplots: {e}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)
