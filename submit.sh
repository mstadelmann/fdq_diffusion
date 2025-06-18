#!/bin/bash

submit_job() {
    root_path="/cluster/home/stmd/dev/fdq_diffusion/configs/"
    python3 /cluster/home/stmd/dev/fonduecaquelon/fdq_submit.py $root_path$1
}

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffuionGenerative_MONAI_Celebrties
#--------------------------------------------------------------------------------------------------

# MONAI
# -----

# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v0.json # ok, but not great, early stop
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v1.json # ok, not much difference to v0
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v2.json # some great, some look like ghosts
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v3.json ----> STILL RUNNING - very slow?
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v4.json # ok
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v5.json # mostly good
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v6.json # okayish
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_d1.json # black
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_d2.json # bad
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_t1.json # bright
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_t2.json # funny
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_t3.json # mostly good
# submit_job celeb_diff_gen_monai/dg_m_cele_x128_t4.json


# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v0.json # -> early stop, ok but not great
# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v4.json # -> early stop 38382
# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v7.json # still running 38383


# CHUCHICHAESTLI
# --------------

# submit_job celeb_diff_gen_cc/dg_c_celeb_x128_v0.json # 38860
# submit_job celeb_diff_gen_cc/dg_c_celeb_x256_v0.json # 38861

#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
# submit_job celeb_vae/vae_celeb_x128.json
# submit_job celeb_vae/vae_celeb_x256.json 38436 # ok
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
submit_job celeb_fm_gen/fm_meta_cele_x128_v0.json




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
# submit_job cbct_vae/vae_cbct_x256_v0.json
# submit_job cbct_vae/vae_cbct_x256_v1.json
# submit_job cbct_vae/vae_cbct_x256_v2.json
# submit_job cbct_vae/vae_cbct_x256_v3.json # 38756
# submit_job cbct_vae/vae_cbct_x256_v4.json

#--------------------------------------------------------------------------------------------------
# CBCT VAE MAESI TEST
#--------------------------------------------------------------------------------------------------

# submit_job cbct_vae/vae_cbct_maisi_x256.json # 38758
submit_job cbct_vae/vae_cbct_maisi_x256_v1.json
submit_job cbct_vae/vae_cbct_maisi_x256_v2.json

#--------------------------------------------------------------------------------------------------
# CBCT LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256.json
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.json

# submit_job cbct_latent_diff/lat_diff_cbct_monai_x256.json