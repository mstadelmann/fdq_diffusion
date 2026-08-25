#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]

#--------------------------------------------------------------------------------------------------
# CBCT VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/cbct_vae/vae_cbct_x64_v0.yaml
fdq_submit configs/cbct_vae/vae_cbct_x64_v0_dist2.yaml       # 2 GPUs

fdq_submit configs/cbct_vae/vae_cbct_x128_v0.yaml            # 128px

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

fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x128_v0.yaml         # MONAI UNet
fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x128_v0_dist2.yaml   # 2 GPUs

fdq_submit configs/cbct_diff_gen/diff_cbct_x128.yaml                  # chuchichaestli UNet

# add 256x !

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
