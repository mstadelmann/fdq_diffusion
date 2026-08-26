# dc_m_ct_x128_deblur - next steps

Current state is a deliberately simple proof of concept: concatenation-conditioned
diffusion, Gaussian blur as a stand-in degradation, no EMA, no guidance. The blur
kernel/sigma mismatch (kernel_size too small to represent the requested sigma) has
already been fixed - everything below is a deliberate simplification to revisit
once the basic pipeline is confirmed to work, not a bug.

## 1. Replace Gaussian blur with a realistic CBCT degradation
Gaussian blur has little in common with real CBCT artifacts (scatter, streaking,
beam hardening, cone-beam geometry). The dataset this config used to train on
(`size_128_debug`, `sca_05`, multiple `nproj` variants, loaded via the now-deleted
`hdf_preparator_v2.py`) used a physically-simulated scatter/limited-projection
degradation - a much better proxy for the actual task. Worth regenerating an
equivalent from the new CT-RATE source if a realistic result matters, rather than
concluding anything about real CBCT correction from the blur-based result.

## 2. Blur is 2D (in-plane), not 3D
`Gaussian_Blur` wraps `torchvision.transforms.GaussianBlur`, which only blurs the
last two tensor dimensions. For this (C, D, H, W) volume that's the axial (H, W)
plane per slice - not a true 3D blur across D too. Fine for a first pass; decide
deliberately whether the more sophisticated setup needs real 3D degradation
(matters more once doing #1 anyway).

## 3. Randomize degradation severity
The degradation (blur, or whatever replaces it per #1) is fixed - identical
strength for every sample. Randomizing it per-sample would let the model
generalize across a range of severities instead of only inverting one specific
one.

## 4. Add EMA
fdq's `ema_decay` (added to `fonduecaquelon` this cycle) is not used here. Likely
the single easiest win for sample quality - add `ema_decay: 0.999` under
`models.monaiUNET`.

## 5. Classifier-free guidance
Condition is always present (plain channel concatenation, no dropout). Randomly
dropping the condition during training, plus guidance-scale weighting at
sampling, is a well-established upgrade over plain concatenation - moderate
effort, meaningful quality gain.

## 6. Larger effective batch size
`train_batch_size: 1`, `accumulate_grad_batches: 1` - very noisy gradients over
what could be ~1.5M optimizer steps (10k samples x 150 epochs). Bumping
`accumulate_grad_batches` to 4-8 is free (no extra activation memory, just delays
the optimizer step) and should stabilize training.

## 7. Track reconstruction quality, not just noise-prediction loss
`losses:` only has MAE/MSE on the diffusion noise prediction. The VAE / VAE-MAISI
configs already define `common.py`'s PSNR/PSNR_hu losses for exactly this kind of
before/after comparison - add them here too, so there's an actual deblurring
quality signal instead of only eyeballing logged slices.

## 8. HU window slightly wastes output range
`HU_to_norm` / `HU_to_norm_blur` clip at `in_min: -1500`, but `preproc_128`
volumes are already floored at -1000 - the bottom third of the [-1, 1] output
range is never used. Same minor issue exists in sibling configs; low priority,
not specific to this one.
