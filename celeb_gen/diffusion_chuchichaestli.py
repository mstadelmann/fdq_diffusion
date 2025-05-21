import torch
import numpy as np
import matplotlib.pyplot as plt
from fdq.misc import print_nb_weights
from fdq.ui_functions import startProgBar, iprint, eprint
from chuchichaestli.diffusion.ddpm import DDPM


# from vlf.img_func import saveImg, createSubplots

# from trainings.diffusion_common import nb_requested_train_inferences


def compute_histo(image, nb_bins=100):
    bins = torch.linspace(image.min(), image.max(), nb_bins).cpu()
    return torch.histogram(image.cpu().float(), bins=bins)


def check_if_image(data):
    """
    Check if tensor is (possibly) an image.
    """
    if data is None:
        return False
    elif len(data.shape) < 2:
        # at least two dimensions
        return False
    # at least 5 pixels in 2 dims to be considered as image
    elif not data.shape[-1] >= 5 or not data.shape[-2] >= 5:
        return False

    return True


def extract_2d_slices(
    images: torch.tensor,
    experiment,
    nb_slices=1,
    select_channel=None,
    squeeze_last=False,
    strict=True,
):
    """
    Extracts 2D slice(s) from tensor.
    Returns list of requested tensors.

    Expected image data format 2D: [H,W], [C,H,W] or [B,C,H,W]
    (Fewer dims in 2D are expected to be singleton dimensions.)
    Expected image data format 3D: [B,C,D,H,W]
    (Fewer dimensions in 3D fill fail.)

    For 3D data, experiment.img_export_dims defines the slicing direction.
    Supported values are  D, W, H or a list of these.

    nb_slices: How many slices to extract. Defaults to 1. Not yet implemented!

    squeeze_last: if true, returns only the last image in the listed
                  as squeezed tensor (for direct plotting or saving).

    strict: if True, raises an error if the image shape is not as expected.

    supported input tensor shapes:
    a)       [H,W] --> [1,1,H,W]
    b)     [C,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
    c)   [B,C,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
                                 or with nb_slices set: [nb_slices,C,H,W]
    d) [B,C,D,H,W] --> [1,C,H,W] or with select_channel set:    [1,1,H,W]
                                 or with nb_slices set: [nb_slices,C,H,W]
                                 or with multiple dimensions set: [[1,C,H,W]]
    """

    if nb_slices != 1:
        raise NotImplementedError("Only one slice is supported at the moment.")

    if select_channel is not None:
        raise NotImplementedError("Selecting a channel is not yet implemented.")

    if not check_if_image(images):
        return []

    image_list = []

    # select center batch img
    batch_size = images.shape[0]
    batch_center = int(batch_size / 2)

    nb_input_dim = len(images.shape)

    # A) [H,W] or B) [C,H,W]
    if nb_input_dim in [2, 3]:
        image_list.append(images)

    # C) [B,C,H,W]
    elif nb_input_dim == 4 and not experiment.is_3d:
        image_list.append(images[batch_center, ...])

    # D) [B,C,D,H,W]
    elif nb_input_dim == 5:
        if not experiment.is_3d:
            raise ValueError("Image data is 3D, but experiment is not 3D.")

        for slicing_dir in experiment.img_export_dims:
            dim = ["B", "C", "D", "H", "W"].index(slicing_dir)
            slice_idx = int(images.shape[dim] / 2)
            if dim == 2:
                image_list.append(images[batch_center, :, slice_idx, :, :])
            elif dim == 3:
                image_list.append(images[batch_center, :, :, slice_idx, :])
            elif dim == 4:
                image_list.append(images[batch_center, :, :, :, slice_idx])

    # 3d experiment, but pre-sliced data
    elif (
        nb_input_dim == 4
        and images.shape[0] == 1
        and images.shape[1] == 1
        and not strict
    ):
        image_list.append(images[0, 0, :, :])

    else:
        raise ValueError(
            f"ERROR, not supported image shape {images.shape}, 3D experiment: {experiment.is_3d}."
        )

    nb_expected_dims = 4

    for i, img in enumerate(image_list):
        while img.dim() < nb_expected_dims:
            img = img.unsqueeze(0)
            image_list[i] = img

    if not squeeze_last:
        return image_list
    else:
        return torch.squeeze(image_list[-1]).cpu()


def createSubplots(
    image_list,
    batch_nb=0,
    labels=None,
    grayscale=False,
    experiment=None,
    histogram=False,
    histogram3d=False,
    save_path=None,
    show_plot=True,
    figure_title=None,
    max_cols=None,
    hide_ticks=False,
    apply_global_range=True,
    lower_percentile_cutoff=0.01,
    upper_percentile_cutoff=0.99,
    fig_size=3,
):
    if labels is None:
        labels = len(image_list) * [""]
    elif len(labels) != len(image_list):
        raise ValueError("ERROR - nb labels does not correspond to nb images!")

    histos_3d = None

    # it might be a 3d experiment, but data was presliced to reduce storage footprint
    reslicing_required = len(image_list[0].shape) > 4

    if experiment is not None and experiment.is_3d and reslicing_required:
        if histogram3d:
            histos_3d = [
                compute_histo(torch.squeeze(img3d).cpu()) for img3d in image_list
            ]

        nb_images = len(image_list) * len(experiment.img_export_dims)
        nb_slices_per_img = len(experiment.img_export_dims)
        image_list = sum(
            [extract_2d_slices(img, experiment, strict=False) for img in image_list], []
        )
        labels = sum(
            [[lbl + "_" + dir for dir in experiment.img_export_dims] for lbl in labels],
            [],
        )

    else:
        nb_images = len(image_list)
        nb_slices_per_img = 1

    # clamper_perc = transformers.get_transformer(
    #     {
    #         "CLAMP_perc": {
    #             "lower_perc": lower_percentile_cutoff,
    #             "upper_perc": upper_percentile_cutoff,
    #         }
    #     }
    # )

    # img_clamp = [clamper_perc(img) for img in image_list]
    img_clamp = image_list
    global_min = min(float(img.min()) for img in img_clamp)
    global_max = max(float(img.max()) for img in img_clamp)

    if histogram3d:
        nb_rows = 3
        nb_cols = nb_images
    elif histogram:
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
        if histogram3d:
            figure_title += " - green = 3D histogram"
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

        if not grayscale and img.dtype == torch.float and img.max() > 50:
            # assume RGB image in float values
            # TODO where is this used? this might not be a good idea!
            img = img.int()

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

        if i % len(experiment.img_export_dims) == 0 and histos_3d is not None:
            hist = histos_3d[int(i / len(experiment.img_export_dims))]
            axs[2, c_col].plot(hist.bin_edges[:-1], hist.hist, color="g")
            # fix_histo_ticks(axs[2, c_col], hist)

            # Make the plot wider?
            if nb_slices_per_img > 1:
                for sl in range(1, nb_slices_per_img):
                    axs[2, c_col + sl].remove()

                axs[2, c_col].set_position(
                    [
                        axs[2, c_col].get_position().x0,
                        axs[2, c_col].get_position().y0,
                        axs[2, c_col].get_position().width * nb_slices_per_img,
                        axs[2, c_col].get_position().height,
                    ]
                )

    if hide_ticks:
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if save_path is not None:
        plt.savefig(save_path)
    elif experiment is not None:
        plt.savefig(experiment.get_next_export_fn())
    else:
        eprint("No save path provided!")

    if show_plot:
        plt.show()


def get_sample_from_noise_cc(experiment, diffuser, shape=None, debug_img_export=False):
    # get train img shape
    if shape is not None:
        gen_shape = shape
    else:
        try:
            if (
                experiment.dataPreparator.n_train > 0
                and experiment.dataPreparator.train_data_loader is not None
            ):
                images_gt = next(iter(experiment.dataPreparator.train_data_loader))[0]
            elif (
                experiment.dataPreparator.n_val > 0
                and experiment.dataPreparator.val_data_loader is not None
            ):
                images_gt = next(iter(experiment.dataPreparator.val_data_loader))[0]
            elif (
                experiment.dataPreparator.n_test > 0
                and experiment.dataPreparator.test_data_loader is not None
            ):
                images_gt = next(iter(experiment.dataPreparator.test_data_loader))[0]
            else:
                raise ValueError(
                    "ERROR - No data loader found. Unable to set image shape."
                )
            gen_shape = images_gt.shape[1:]

        except StopIteration as e:
            # next() above does not handle the stop iteration, thus we fix it here
            raise ValueError("ERROR - No data found. Unable to set image shape.") from e

    nb_timesteps = experiment.train_params.get("diffusion_nb_steps")
    nb_plot_steps = experiment.expfile.get("store", {}).get("nb_plot_steps", 5)

    img = None

    with torch.no_grad():
        experiment.networkModel.eval()

        while True:
            intermediate_imgs = []
            idx_to_store = torch.linspace(
                0, nb_timesteps - 1, nb_plot_steps, dtype=torch.int
            ).tolist()

            for j, img in enumerate(
                diffuser.generate(
                    model=experiment.networkModel,
                    shape=tuple(gen_shape),
                    n=1,
                    yield_intermediate=True,
                )
            ):
                if j in idx_to_store:
                    intermediate_imgs.append(img)

            createSubplots(
                image_list=intermediate_imgs,
                grayscale=True,
                experiment=experiment,
                histogram=True,
                histogram3d=True,
                figure_title="Generative Diffusion Steps",
                labels=[f"Step {i}" for i in idx_to_store],
            )

            # if debug_img_export and img is not None:
            #     fn = experiment.get_next_export_fn(name="generated", file_ending="jpg")
            #     saveImg(img=img, path=fn, experiment=experiment)

            yield intermediate_imgs[-1]


def train(experiment) -> None:
    iprint("Chuchichaestli Diffusion Training")
    print_nb_weights(experiment)

    data = experiment.data["celeb_HDF"]
    model = experiment.models["ccUNET"]

    train_loader = data.train_data_loader

    # nb_plot_steps = min(experiment.expfile.get("store", {}).get("nb_plot_steps", 10), 15)

    chuchi_diffuser = DDPM(
        num_timesteps=experiment.exp_def.train.args.diffusion_nb_steps,
        device=experiment.device,
        beta_start=experiment.exp_def.train.args.diffusion_shd_beta_start,
        beta_end=experiment.exp_def.train.args.diffusion_shd_beta_end,
        schedule=experiment.exp_def.train.args.diffusion_scheduler,
    )

    for epoch in range(experiment.start_epoch, experiment.nb_epochs):
        experiment.current_epoch = epoch
        iprint(f"\nEpoch: {epoch + 1} / {experiment.nb_epochs}")

        model.train()
        train_loss_sum = 0.0
        pbar = startProgBar(data.n_train_batches, "training...")

        for nb_tbatch, batch in enumerate(train_loader):
            pbar.update(nb_tbatch)
            images_gt = batch[0].to(experiment.device)

            # if nb_tbatch == 0 and epoch in (0, experiment.start_epoch):
            #     # run dummy forward pass to visualize noise schedule
            #     # TODO

            #     # Plot diffusion schedule
            #     # TODO

            #     # first test batch: store inputs
            #     fn = experiment.get_next_export_fn(name="input_gt", file_ending="png")
            #     # saveImg(img=images_gt, path=fn, experiment=experiment)
            #     createSubplots(
            #         image_list=[images_gt],
            #         grayscale=True,
            #         experiment=experiment,
            #         histogram=True,
            #         histogram3d=True,
            #         save_path=fn,
            #         figure_title="Input GT",
            #     )

            #     if experiment.net_input_size is not None:
            #         raise ValueError(
            #             "ERROR - net_input_size was manually defined in experiment file."
            #         )

            #     # ignore batch size
            #     experiment.net_input_size = [1] + list(images_gt.shape[1:])

            # this can be written without code repetition, however, goal is to keep both options flexible...
            # following: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            if experiment.useAMP:
                device_type = (
                    "cuda" if experiment.device == torch.device("cuda") else "cpu"
                )

                with torch.autocast(device_type=device_type, enabled=True):
                    noisy_imgs, noise, t = chuchi_diffuser.noise_step(images_gt)
                    noise_pred = model(noisy_imgs, t)
                    train_loss_tensor = (
                        experiment.losses["MAE"](noise, noise_pred)
                        / experiment.gradacc_iter
                    )

                # noise_pred becomes NaN if AMP is used. Why?
                # https://pytorch.org/docs/stable/amp.html
                # https://pytorch.org/docs/2.2/notes/amp_examples.html#amp-examples

                experiment.scaler.scale(train_loss_tensor).backward()

            else:
                noisy_imgs, noise, t = chuchi_diffuser.noise_step(images_gt)
                noise_pred = model(noisy_imgs, t)
                train_loss_tensor = (
                    experiment.losses["MAE"](noise, noise_pred)
                    / experiment.gradacc_iter
                )

                train_loss_tensor.backward()

            experiment.update_gradients(
                b_idx=nb_tbatch, loader_name="celeb_HDF", model_name="ccUNET"
            )

            train_loss_sum += train_loss_tensor.detach().item()

            # if math.isnan(train_loss_sum):
            #     print("NAN Loss detected. Skipping batch.")

        pbar.finish()

        # if experiment.useWandb:
        #     save_wandb_loss(experiment)

        # for _ in range(nb_requested_train_inferences(experiment)):
        #     # Dummy validation: generate samples
        #     img = next(
        #         get_sample_from_noise_cc(
        #             experiment=experiment,
        #             diffuser=chuchi_diffuser,
        #             shape=experiment.net_input_size[1:],
        #             debug_img_export=True,
        #         )
        #     )

        #     if experiment.useWandb:
        #         save_wandb(experiment, [("sample result", img)])

        experiment.trainLoss = train_loss_sum / len(train_loader.dataset)
        experiment.valLoss = 0  # no validation train_loss_tensor

        iprint(f"Training Loss: {experiment.trainLoss:.4f}")

        experiment.finalize_epoch()

        if experiment.check_early_stop():
            break
