def augment(sample, experiment=None):
    """Apply a random horizontal flip to a cached celeb image sample.

    Cached samples have already passed through the deterministic 'resize_norm'
    transform once, at cache-build time. This runs on every access instead, so
    the flip decision is resampled every epoch rather than baked in once.

    Args:
        sample: A single cached image tensor (C, H, W).
        experiment: The main FDQ experiment object, providing access to
                    registered transformers.

    Returns:
        torch.Tensor: The (possibly flipped) image tensor.
    """
    return experiment.transformers["random_h_flip"](sample)
