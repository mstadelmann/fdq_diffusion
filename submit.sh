#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml           # A100: 33s/ep
fdq_submit configs/celeb_vae/vae_celeb_x128_03_dist2.yaml          # A100: 19s/ep

fdq_submit configs/celeb_vae/vae_celeb_x256_00.yaml                # H100: 1m 30s/ep

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
# dg_c_cele_ = DiffusionGenerative_Chuchichaestli_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml         # baseline, 2m40/ep
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base_dist2.yaml   # 1m40/ep
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml              # res-block off
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml              # 3 att levels
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml              # 4 layers only
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_08_zero_snr.yaml


fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_00.yaml
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_01_zero_snr.yaml


# CHUCHICHAESTLI
# --------------
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml        # LR param
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml        # lin/cos/sig param

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_00_base.yaml
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x256_00.yaml


#--------------------------------------------------------------------------------------------------
# CELEB Flow Matching Generative
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v0.yaml
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v1.yaml
fdq_submit configs/celeb_fm_gen/fm_meta_cele_x128_v2.yaml

fdq_submit configs/celeb_fm_gen/fm_meta_cele_x256_v0.yaml

##################################################################################################
##################################################################################################

#--------------------------------------------------------------------------------------------------
# CBCT VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/cbct_vae/vae_cbct_x64_v0.yaml
fdq_submit configs/cbct_vae/vae_cbct_x64_v0_dist2.yaml

fdq_submit configs/cbct_vae/vae_cbct_x128_v0.yaml

# add x256

#--------------------------------------------------------------------------------------------------
# CBCT VAE MAESI TEST
#--------------------------------------------------------------------------------------------------

fdq_submit configs/cbct_vae/vae_cbct_maisi_x128.yaml

# fdq_submit configs/cbct_vae/vae_cbct_maisi_x256.yaml
# fdq_submit configs/cbct_vae/vae_cbct_maisi_x256_v1.yaml
# fdq_submit configs/cbct_vae/vae_cbct_maisi_x256_v2.yaml # 42101 -> best


#--------------------------------------------------------------------------------------------------
# CBCT DIFF GEN
#--------------------------------------------------------------------------------------------------

fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x128_v0.yaml

# fdq_submit configs/cbct_diff_gen/diff_cbct_x256.yaml # -> noisy
# fdq_submit configs/cbct_diff_gen/diff_cbct_x256_minmax_norm.yaml
# fdq_submit configs/cbct_diff_gen/diff_cbct_x256_sig1000.yaml
# fdq_submit configs/cbct_diff_gen/diff_cbct_x256_attv1.yaml
# fdq_submit configs/cbct_diff_gen/diff_cbct_x256_attv2.yaml
# fdq_submit configs/cbct_diff_gen/diff_cbct_x256_01norm.yaml

#--------------------------------------------------------------------------------------------------
# CBCT DIFF Cnd
#--------------------------------------------------------------------------------------------------
# blur
# fdq_submit configs/cbct_diff_cond_monai/dc_m_cbct_x128_v0.yaml

# cbct artifact
# fdq_submit configs/cbct_diff_cond_monai/dc_m_cbct_x128_v1.yaml
# test: 38420
# resume: 38421

# fdq_submit configs/cbct_diff_cond_monai/dc_m_cbct_x256_v0.yaml
# 38319



#--------------------------------------------------------------------------------------------------
# CBCT LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_maisi_x256.yaml
# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.yaml

# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_monai_x256.yaml