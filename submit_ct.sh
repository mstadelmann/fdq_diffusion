#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]

#--------------------------------------------------------------------------------------------------
# CT-Rate VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/cbct_vae/vae_cbct_x64_v0.yaml            # baseline/memory check only
fdq_submit configs/cbct_vae/vae_cbct_x64_v0_dist2.yaml

fdq_submit configs/cbct_vae/vae_cbct_x128_v0.yaml

fdq_submit configs/cbct_vae/vae_cbct_x256_v0_dist2.yaml     # increased sys mem: 58121, nncl timeout, restart 58156

#--------------------------------------------------------------------------------------------------
# CT-Rate  VAE MAESI TEST
#--------------------------------------------------------------------------------------------------
fdq_submit configs/cbct_vae/vae_cbct_maisi_x128.yaml

fdq_submit configs/cbct_vae/vae_cbct_maisi_x256.yaml
fdq_submit configs/cbct_vae/vae_cbct_maisi_x256_v1.yaml
fdq_submit configs/cbct_vae/vae_cbct_maisi_x256_v2.yaml


#--------------------------------------------------------------------------------------------------
# CT-Rate  DIFF GEN
#--------------------------------------------------------------------------------------------------
# TODO 58158 - 63
fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x128_v0.yaml         # MONAI UNet, baseline/memory check only
fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x128_v0_dist2.yaml
fdq_submit configs/cbct_diff_gen_cc/dg_cc_ct_x128.yaml                # chuchichaestli UNet

fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x256_v0.yaml         # MONAI UNet
fdq_submit configs/cbct_diff_gen_monai/dg_m_cbct_x256_v0_dist2.yaml
fdq_submit configs/cbct_diff_gen_cc/dg_cc_ct_x256.yaml                # chuchichaestli UNet

#--------------------------------------------------------------------------------------------------
# CT-Rate  DIFF Cnd
#--------------------------------------------------------------------------------------------------
# blur
fdq_submit configs/cbct_diff_cond_monai/dc_m_ct_x64_deblur.yaml     # MONAI UNet
fdq_submit configs/cbct_diff_cond_monai/dc_m_ct_x128_deblur.yaml

fdq_submit configs/cbct_diff_cond_cc/dc_cc_ct_x64_deblur.yaml       # chuchichaestli UNet


#--------------------------------------------------------------------------------------------------
# CT-Rate  LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_maisi_x256.yaml
# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.yaml

# fdq_submit configs/cbct_latent_diff/lat_diff_cbct_monai_x256.yaml
