#!/bin/bash

# Preparations:
# 1) create/enable PY env
# 2) pip install fdq[submit]


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml # 18464, 250ep, 4h31
fdq_submit configs/celeb_vae/vae_celeb_x128_02_caching_nbw4.yaml
fdq_submit configs/celeb_vae/vae_celeb_x128_03_dist2.yaml # 18467, 250ep, 3h01
fdq_submit configs/celeb_vae/vae_celeb_x128_03c_dist2_more_kl.yaml
fdq_submit configs/celeb_vae/vae_celeb_x128_03d_dist2_less_kl.yaml

# fdq_submit configs/celeb_vae/vae_celeb_x128_01_caching_dist2.yaml # 18465, 250ep, 4h39
# fdq_submit configs/celeb_vae/vae_celeb_x128_02_caching_nbw0.yaml # 18466, 250ep, 5h40
fdq_submit configs/celeb_vae/vae_celeb_x128_03b_dist2_caching.yaml # 18664, 250ep, 5h13 -> crashed at the end, try again!
# fdq_submit configs/celeb_vae/vae_celeb_x128_04_small.yaml
# --> small is the worst, the others are qualitatively very similar!


fdq_submit configs/celeb_vae/vae_celeb_x256_00_dist2.yaml # 18666, 315ep 7h43


#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffusionGenerative_MONAI_Celebrities
#--------------------------------------------------------------------------------------------------

# MONAI
# -----
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_00_base.yaml # baseline
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_01.yaml # res-block off
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_02.yaml # 3 att levels
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_03.yaml # 4 layers only

# play with data norm
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_04.yaml
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_05.yaml
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_06.yaml
# fdq_submit configs/celeb_diff_gen_monai/dg_m_cele_x128_07.yaml
# --> did not help!


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


# CHUCHICHAESTLI
# --------------

# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_00_base.yaml
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_01.yaml
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_02.yaml
fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_03_dist2.yaml

# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_00_base.yaml

# DDP not currently working: AttributeError: Can't pickle local object 'FCQmode._create_setter.<locals>.setter'
# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x128_v0_dist2.json 
# fdq_submit configs/celeb_diff_gen_cc/dg_c_celeb_x256_v0.json #



#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_00_base.yaml # 18667
fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x128_01_small.yaml # 18668, 86 EP, 1h10


fdq_submit configs/celeb_latent_diff/lat_diff_celeb_x256_00.yaml

# fdq_submit celeb_Ldiff_gen_monai/dgL_m_cele_x256_v0.json


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