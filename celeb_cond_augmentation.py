def augment(sample, experiment=None):
    """Apply the registered 'cond_random' transform to a cached (target, condition) pair.

    Cached samples have already passed through their deterministic transform once, at
    cache-build time. This runs on every access instead, so any randomness in
    'cond_random' (cutout, noise, half-face masking, ...) is resampled every epoch
    rather than baked in once. Only the condition (second element) is corrupted - the
    target passes through unchanged.

    Args:
        sample: A (target, condition) tensor tuple.
        experiment: The main FDQ experiment object, providing access to registered
                    transformers.

    Returns:
        tuple: (target, corrupted_condition).
    """
    target, condition = sample
    return target, experiment.transformers["cond_random"](condition)
