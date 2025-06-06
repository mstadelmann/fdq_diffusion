import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def compute_histo(image, nb_bins=100):
    bins = torch.linspace(image.min(), image.max(), nb_bins).cpu()
    return torch.histogram(image.cpu().float(), bins=bins)


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

    try:

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
                    axs[1, i].set_yticks([])
                    axs[1, i].set_yticklabels([])

        if save_path is None:
            save_path = experiment.get_next_export_fn()

        plt.savefig(save_path)

        return save_path

    except Exception as e:
        print(f"An error occurred while creating subplots: {e}")
        return None
