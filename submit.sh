#!/bin/bash

submit_job() {
    root_path="/cluster/home/stmd/dev/fdq_diffusion/configs/"
    python3 /cluster/home/stmd/dev/fonduecaquelon/fdq_submit.py $root_path$1
    sleep 5
}

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffuionGenerative_MONAI_Celebrties
#--------------------------------------------------------------------------------------------------

# MONAI
# -----

submit_job celeb_diff_gen_monai/dg_m_cele_x128_v0.json # baseline
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v1.json # res-block off
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v2.json # 3 att levels
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v3.json # all att layers - very slow; try on hopper!
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v4.json # larger net, two att
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v5.json # deeper net, one att
submit_job celeb_diff_gen_monai/dg_m_cele_x128_v6.json # more att head channels
submit_job celeb_diff_gen_monai/dg_m_cele_x128_d1.json # [0,1] norm
submit_job celeb_diff_gen_monai/dg_m_cele_x128_d2.json # random flipping
submit_job celeb_diff_gen_monai/dg_m_cele_x128_t1.json # 1000 steps
submit_job celeb_diff_gen_monai/dg_m_cele_x128_t2.json # sigmoid beta
submit_job celeb_diff_gen_monai/dg_m_cele_x128_t3.json # smaller LR
submit_job celeb_diff_gen_monai/dg_m_cele_x128_t4.json # linear beta


submit_job celeb_diff_gen_monai/dg_m_cele_x256_v0.json #
submit_job celeb_diff_gen_monai/dg_m_cele_x256_v4.json #
submit_job celeb_diff_gen_monai/dg_m_cele_x256_v7.json #


# CHUCHICHAESTLI
# --------------

submit_job celeb_diff_gen_cc/dg_c_celeb_x128_v0.json #

# DDP not currently working: AttributeError: Can't pickle local object 'FCQmode._create_setter.<locals>.setter'
# submit_job celeb_diff_gen_cc/dg_c_celeb_x128_v0_dist2.json 

submit_job celeb_diff_gen_cc/dg_c_celeb_x256_v0.json #


#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
submit_job celeb_vae/vae_celeb_x128.json 
# submit_job celeb_vae/vae_celeb_x128_dist4_nbw0.json 
# submit_job celeb_vae/vae_celeb_x128_dist2_nbw0.json 
# submit_job celeb_vae/vae_celeb_x128_dist2_nbw4.json 
# submit_job celeb_vae/vae_celeb_x128_nbw0.json 

submit_job celeb_vae/vae_celeb_x256.json #
# submit_job celeb_vae/vae_celeb_x256_more_kl.json # -> this results in poor results (mu nicely centered at 0, but img quality bad)
# submit_job celeb_vae/vae_celeb_x256_less_kl.json


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
# submit_job celeb_latent_diff/lat_diff_celeb_x128.json
# submit_job celeb_latent_diff/lat_diff_celeb_x256.json

# submit_job celeb_Ldiff_gen_monai/dgL_m_cele_x256_v0.json


#--------------------------------------------------------------------------------------------------
# CELEB Flow Matching Generative
#--------------------------------------------------------------------------------------------------
# submit_job celeb_fm_gen/fm_meta_cele_x128_v0.json # test is missing! -> fixed
# submit_job celeb_fm_gen/fm_meta_cele_x128_v0_dist2.json # error num workers and HDF (42095)! currently running with nworks=0 (42472)



# submit_job celeb_fm_gen/fm_meta_cele_x256_v0.json
# submit_job celeb_fm_gen/fm_meta_cele_x256_v1.json

##################################################################################################
##################################################################################################



#--------------------------------------------------------------------------------------------------
# CBCT DIFF GEN
#--------------------------------------------------------------------------------------------------

# submit_job cbct_diff_gen_monai/dg_m_cbct_x128_v0.json

# submit_job cbct_diff_gen/diff_cbct_x256.json # -> noisy
# submit_job cbct_diff_gen/diff_cbct_x256_minmax_norm.json
# submit_job cbct_diff_gen/diff_cbct_x256_sig1000.json
# submit_job cbct_diff_gen/diff_cbct_x256_attv1.json
# submit_job cbct_diff_gen/diff_cbct_x256_attv2.json
# submit_job cbct_diff_gen/diff_cbct_x256_01norm.json

# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v0.json
# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v1.json
# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v2.json
# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v3.json

#--------------------------------------------------------------------------------------------------
# CBCT DIFF Cnd
#--------------------------------------------------------------------------------------------------
# blur
# submit_job cbct_diff_cond_monai/dc_m_cbct_x128_v0.json

# cbct artifact
# submit_job cbct_diff_cond_monai/dc_m_cbct_x128_v1.json
# test: 38420
# resume: 38421

# submit_job cbct_diff_cond_monai/dc_m_cbct_x256_v0.json
# 38319


#--------------------------------------------------------------------------------------------------
# CBCT VAE
#--------------------------------------------------------------------------------------------------
# submit_job cbct_vae/vae_cbct_x256_v0.json # OK (42096)
# submit_job cbct_vae/vae_cbct_x256_v0_dist2.json 42564
# submit_job cbct_vae/vae_cbct_x256_v0_dist3.json 42565

# submit_job cbct_vae/vae_cbct_x256_v1.json
# submit_job cbct_vae/vae_cbct_x256_v2.json
# submit_job cbct_vae/vae_cbct_x256_v3.json # 38756
# submit_job cbct_vae/vae_cbct_x256_v4.json

#--------------------------------------------------------------------------------------------------
# CBCT VAE MAESI TEST
#--------------------------------------------------------------------------------------------------

# submit_job cbct_vae/vae_cbct_maisi_x256.json # 38758
# submit_job cbct_vae/vae_cbct_maisi_x256_v1.json
# submit_job cbct_vae/vae_cbct_maisi_x256_v2.json # 42101 -> best

# submit_job cbct_vae/vae_cbct_maisi_x512.json # 42102
# submit_job cbct_vae/vae_cbct_maisi_x512_v2.json # 42569

#--------------------------------------------------------------------------------------------------
# CBCT LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256.json
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.json

# submit_job cbct_latent_diff/lat_diff_cbct_monai_x256.json