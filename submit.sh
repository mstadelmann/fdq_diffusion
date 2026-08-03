#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
# fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml           # A100: 33s/ep
# fdq_submit configs/celeb_vae/vae_celeb_x128_03_dist2.yaml          # A100: 19s/ep

# fdq_submit configs/celeb_vae/vae_celeb_x256_00.yaml                # H100: 1m 30s/ep

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml # baseline, 2m40/ep
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base_dist2.yaml # 1m40/ep
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml # res-block off
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml # 3 att levels
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml # 4 layers only


# play with data norm
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_04.yaml
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_05.yaml # error 39815, resubmitted as 40098
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_06.yaml
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_07.yaml
# --> did not help!

# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_08_zero_snr.yaml # 40112 stopped after 118ep, why? run again: 41399


# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_v4.json # larger net, two att
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_v5.json # deeper net, one att
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_v6.json # more att head channels
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_d1.json # [0,1] norm
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_d2.json # random flipping
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_t1.json # 1000 steps
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_t2.json # sigmoid beta
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_t3.json # smaller LR
# fdq_submit celeb_diff_gen_monai/dg_m_cele_x128_t4.json # linear beta


# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_00.yaml
fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x256_01_zero_snr.yaml


# CHUCHICHAESTLI
# --------------

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml
# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml # LR param
# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml # lin/cos/sig param

fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_00_base.yaml # train samples good, test poor
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x256_00.yaml # train samples good, test poor


#--------------------------------------------------------------------------------------------------
# CELEB Flow Matching Generative
#--------------------------------------------------------------------------------------------------
# fdq_submit celeb_fm_gen/fm_meta_cele_x128_v0.json # test is missing! -> fixed
# fdq_submit celeb_fm_gen/fm_meta_cele_x128_v0_dist2.json # error num workers and HDF (42095)! currently running with nworks=0 (42472)



# fdq_submit celeb_fm_gen/fm_meta_cele_x256_v0.json
# fdq_submit celeb_fm_gen/fm_meta_cele_x256_v1.json

##################################################################################################
##################################################################################################



#--------------------------------------------------------------------------------------------------
# CBCT DIFF GEN
#--------------------------------------------------------------------------------------------------

# fdq_submit cbct_diff_gen_monai/dg_m_cbct_x128_v0.json

# fdq_submit cbct_diff_gen/diff_cbct_x256.json # -> noisy
# fdq_submit cbct_diff_gen/diff_cbct_x256_minmax_norm.json
# fdq_submit cbct_diff_gen/diff_cbct_x256_sig1000.json
# fdq_submit cbct_diff_gen/diff_cbct_x256_attv1.json
# fdq_submit cbct_diff_gen/diff_cbct_x256_attv2.json
# fdq_submit cbct_diff_gen/diff_cbct_x256_01norm.json

# fdq_submit cbct_diff_gen_monai/dg_m_cbct_x256_v0.json
# fdq_submit cbct_diff_gen_monai/dg_m_cbct_x256_v1.json
# fdq_submit cbct_diff_gen_monai/dg_m_cbct_x256_v2.json
# fdq_submit cbct_diff_gen_monai/dg_m_cbct_x256_v3.json

#--------------------------------------------------------------------------------------------------
# CBCT DIFF Cnd
#--------------------------------------------------------------------------------------------------
# blur
# fdq_submit cbct_diff_cond_monai/dc_m_cbct_x128_v0.json

# cbct artifact
# fdq_submit cbct_diff_cond_monai/dc_m_cbct_x128_v1.json
# test: 38420
# resume: 38421

# fdq_submit cbct_diff_cond_monai/dc_m_cbct_x256_v0.json
# 38319


#--------------------------------------------------------------------------------------------------
# CBCT VAE
#--------------------------------------------------------------------------------------------------
# fdq_submit cbct_vae/vae_cbct_x256_v0.json # OK (42096)
# fdq_submit cbct_vae/vae_cbct_x256_v0_dist2.json 42564
# fdq_submit cbct_vae/vae_cbct_x256_v0_dist3.json 42565

# fdq_submit cbct_vae/vae_cbct_x256_v1.json
# fdq_submit cbct_vae/vae_cbct_x256_v2.json
# fdq_submit cbct_vae/vae_cbct_x256_v3.json # 38756
# fdq_submit cbct_vae/vae_cbct_x256_v4.json

#--------------------------------------------------------------------------------------------------
# CBCT VAE MAESI TEST
#--------------------------------------------------------------------------------------------------

# fdq_submit cbct_vae/vae_cbct_maisi_x256.json # 38758
# fdq_submit cbct_vae/vae_cbct_maisi_x256_v1.json
# fdq_submit cbct_vae/vae_cbct_maisi_x256_v2.json # 42101 -> best

# fdq_submit cbct_vae/vae_cbct_maisi_x512.json # 42102
# fdq_submit cbct_vae/vae_cbct_maisi_x512_v2.json # 42569

#--------------------------------------------------------------------------------------------------
# CBCT LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# fdq_submit cbct_latent_diff/lat_diff_cbct_maisi_x256.json
# fdq_submit cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.json

# fdq_submit cbct_latent_diff/lat_diff_cbct_monai_x256.json