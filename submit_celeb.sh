#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml           # A100: 33s/ep, baseline/memory check only
fdq_submit configs/celeb_vae/vae_celeb_x128_00_dist2.yaml          # A100: 19s/ep

fdq_submit configs/celeb_vae/vae_celeb_x256_00.yaml                # H100: 1m 30s/ep

fdq_submit configs/celeb_vae/vae_celeb_x512_00.yaml                # H100, same batch_size as x256 - untested, may need reducing

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
# dg_c_cele_ = DiffusionGenerative_Chuchichaestli_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
# UNet from Project MONAI (medical imaging DL toolkit)
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml         # 2m40/ep, baseline/memory check only
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base_dist2.yaml   # 1m40/ep
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml              # res-block off
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml              # 3 att levels
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml              # 4 layers only
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_08_zero_snr.yaml     # zero-SNR fix, v-pred


fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_00.yaml              # 256px, 2 GPUs
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_01_zero_snr.yaml     # 256px, zero-SNR fix


# CHUCHICHAESTLI
# --------------
# UNet from chuchichaestli, ZHAW's own diffusion library
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml   # baseline
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml        # LR param
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml        # lin/cos/sig param

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml   # 256px


# VIT (DiT)
# ---------
# dg_vit_celeb_ = DiffusionGenerative_VisionTransformer_Celebrities
# transformer backbone instead of a UNet (Diffusion Transformer / Peebles & Xie)
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base.yaml         # 1 GPU, baseline/memory check only
# fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base_resume.yaml
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base_dist2.yaml   # 2 GPUs

fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x256_00_base.yaml         # 256px, 2 GPUs


#--------------------------------------------------------------------------------------------------
# CELEB DIFF COND
# blurred GT as condition, same task as CT's cbct_diff_cond_monai/cbct_diff_cond_cc
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_diff_cond_monai/dc_m_celeb_x128_deblur.yaml   # MONAI UNet
fdq_submit configs/celeb_diff_cond_cc/dc_cc_celeb_x128_deblur.yaml     # chuchichaestli UNet


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
# diffusion runs on VAE latents instead of raw pixels - smaller, faster to train
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_00_base.yaml   # UNet backbone
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x256_00.yaml        # 256px

# VIT (DiT) backbone, same pretrained VAEs
fdq_submit configs/celeb_latent_diff_vit/lat_diff_vit_celeb_x128_00_base.yaml
fdq_submit configs/celeb_latent_diff_vit/lat_diff_vit_celeb_x256_00.yaml     # 256px


#--------------------------------------------------------------------------------------------------
# CELEB Flow Matching Generative
# regresses a straight-line velocity field noise->image, no fixed noise schedule
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v0.yaml   # baseline
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v1.yaml   # smaller batch
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v2.yaml   # EMA, attn tweaks

fdq_submit configs/celeb_fm_gen/fm_meta_cele_x256_v0.yaml   # 256px
