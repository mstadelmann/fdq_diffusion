#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml           # quick single-GPU smoke test
fdq_submit configs/celeb_vae/vae_celeb_x128_00_dist2.yaml          # 128px, 2 GPUs
fdq_submit configs/celeb_vae/vae_celeb_x256_00.yaml                # 256px
fdq_submit configs/celeb_vae/vae_celeb_x512_00.yaml                # 512px

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
# dg_c_cele_ = DiffusionGenerative_Chuchichaestli_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
# UNet from Project MONAI (medical imaging DL toolkit)
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml         # quick single-GPU smoke test
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base_dist2.yaml   # baseline, 2 GPUs # TODO: restarted because exp below was better, why? -> 57565
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml              # resblock_updown: false
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml              # 3 att levels
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml              # 4 layers only
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_08_zero_snr.yaml     # zero-SNR fix, v-pred


fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_00.yaml              # 256px, 2 GPUs
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_01_zero_snr.yaml     # 256px, zero-SNR fix


# CHUCHICHAESTLI
# --------------
# UNet from chuchichaestli, ZHAW's own diffusion library
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml   # baseline # TODO, rerun with early stop at 150 -> 57588
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml        # LR sweep, 4 parallel runs
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml        # lin/cos/sig noise-schedule sweep, 3 parallel runs

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml   # 256px -> TODO rerun with early stop at 150: 57592


# VIT (DiT)
# ---------
# dg_vit_celeb_ = DiffusionGenerative_VisionTransformer_Celebrities
# transformer backbone instead of a UNet (Diffusion Transformer / Peebles & Xie)
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base.yaml         # quick single-GPU smoke test
fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x128_00_base_dist2.yaml   # baseline, 2 GPUs

fdq_submit configs/celeb_diff_gen_vit/dg_vit_celeb_x256_00_base.yaml         # 256px, 2 GPUs


#--------------------------------------------------------------------------------------------------
# CELEB DIFF COND
# blurred/corrupted GT as condition, same task as CT's cbct_diff_cond_monai/cbct_diff_cond_cc
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_diff_cond_monai/dc_m_celeb_x128_deblur.yaml   # MONAI UNet
fdq_submit configs/celeb_diff_cond_monai/dc_m_celeb_x128_hard.yaml     # harder condition: blur + cutout + salt-pepper + gaussian noise
fdq_submit configs/celeb_diff_cond_monai/dc_m_celeb_x128_halfface.yaml # condition is half the face blacked out - outpaint the other half

fdq_submit configs/celeb_diff_cond_cc/dc_cc_celeb_x128_deblur.yaml     # chuchichaestli UNet
fdq_submit configs/celeb_diff_cond_cc/dc_cc_celeb_x128_hard.yaml       # harder condition: blur + cutout + salt-pepper + gaussian noise
fdq_submit configs/celeb_diff_cond_cc/dc_cc_celeb_x128_halfface.yaml   # condition is half the face blacked out - outpaint the other half


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
