#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml           # A100: 10 EP, 24min, 1m31s/EP
fdq_submit configs/celeb_vae/vae_celeb_x128_00_dist2.yaml          # A100: 365 EP, 5h4m, 54s/EP

fdq_submit configs/celeb_vae/vae_celeb_x256_00.yaml                # H100: 223 EP, 17h51m, 7m32s/EP

fdq_submit configs/celeb_vae/vae_celeb_x512_00.yaml                # H200, OOM Error, restarted as 50835

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
# dg_c_cele_ = DiffusionGenerative_Chuchichaestli_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
# UNet from Project MONAI (medical imaging DL toolkit)
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml         # 10 EP, 1h17m, 6m54s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base_dist2.yaml   # 65 EP, 4h16m, 3m48s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml              # res-block off, 183 EP, 10h36m, 3m22s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml              # 3 att levels, 153 EP, 19h58m, 7m42s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml              # 4 layers only, 85 EP, 9h51m, 6m52s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_08_zero_snr.yaml     # zero-SNR fix, v-pred, 96 EP, 6h16m, 3m46s/EP


fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_00.yaml              # 256px, 2 GPUs, 149 EP, 40h35m, 13m35s/EP
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_01_zero_snr.yaml     # 256px, zero-SNR fix, 109 EP, 11h2m, 6m0s/EP


# CHUCHICHAESTLI
# --------------
# UNet from chuchichaestli, ZHAW's own diffusion library
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml   # baseline, 165 EP, 2h41m, 1m0s/EP
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml        # LR param, 4 runs (116-178 EP), up to 2h55m, ~57s/EP
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml        # lin/cos/sig param, 3 runs (172-216 EP), up to 3h31m, ~54s/EP

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml   # 256px, 130 EP, 2h29m, 1m5s/EP


# VIT (DiT)
# ---------
# dg_vit_celeb_ = DiffusionGenerative_VisionTransformer_Celebrities
# transformer backbone instead of a UNet (Diffusion Transformer / Peebles & Xie)
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base.yaml         # 1 GPU, 10 EP, 3h14m, 18m56s/EP
# fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base_resume.yaml
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base_dist2.yaml   # 2 GPUs, 350 EP, 56h33m, 9m32s/EP

fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x256_00_base.yaml         # 256px, 2 GPUs, 302 EP, 35h16m, 10m50s/EP


#--------------------------------------------------------------------------------------------------
# CELEB DIFF COND
# blurred GT as condition, same task as CT's cbct_diff_cond_monai/cbct_diff_cond_cc
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_diff_cond_monai/dc_m_celeb_x128_deblur.yaml   # MONAI UNet - EXPERIMENT FAILED (tensor size mismatch) -> resubmitted as 50864
fdq_submit configs/celeb_diff_cond_cc/dc_cc_celeb_x128_deblur.yaml     # chuchichaestli UNet, 128 EP, 2h10m, 55s/EP -> resubmitted as 50873


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
# diffusion runs on VAE latents instead of raw pixels - smaller, faster to train
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_00_base.yaml   # UNet backbone - EXPERIMENT FAILED (KeyError 'MSE') -> 50875
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x256_00.yaml        # 256px - EXPERIMENT FAILED (KeyError 'MSE') -> 50876

# VIT (DiT) backbone, same pretrained VAEs
fdq_submit configs/celeb_latent_diff_vit/lat_diff_vit_celeb_x128_00_base.yaml   # 400 EP, 13h27m, 2m2s/EP
fdq_submit configs/celeb_latent_diff_vit/lat_diff_vit_celeb_x256_00.yaml     # 256px, 400 EP, 52h29m, 7m47s/EP


#--------------------------------------------------------------------------------------------------
# CELEB Flow Matching Generative
# regresses a straight-line velocity field noise->image, no fixed noise schedule
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v0.yaml   # baseline - EXPERIMENT FAILED (DDP SIGABRT on startup) -> 50884
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v1.yaml   # smaller batch, ~408 EP in 23h3m (3m20s/EP) - EXPERIMENT FAILED (DDP SIGABRT on resume) -> 50885
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v2.yaml   # EMA, attn tweaks, ~130 EP in 22h59m (9m54s/EP) - EXPERIMENT FAILED (DDP SIGABRT on resume) -> 50886

fdq_submit configs/celeb_fm_gen/fm_meta_cele_x256_v0.yaml   # 256px, ~16 EP over ~42h (160m/EP) - EXPERIMENT FAILED (DDP SIGABRT on resume) -> 50887
