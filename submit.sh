#!/bin/bash

submit_job() {
    root_path="/cluster/home/stmd/dev/fdq_diffusion/configs/"
    python3 /cluster/home/stmd/dev/fonduecaquelon/fdq_submit.py $root_path$1
}

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
# dg_m_cele_ = DiffuionGenerative_MONAI_Celebrties
#--------------------------------------------------------------------------------------------------

# submit_job celeb_diff_gen_monai/dg_m_cele_x128_v0.json # ok, but not great
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


# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v0.json # -> early stop
# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v4.json # -> early stop
# submit_job celeb_diff_gen_monai/dg_m_cele_x256_v7.json # still running


# old

# submit_job celeb_diff_gen/diff_celeb_x128_small.json # model 1.37M = 5MB -> bad
# submit_job celeb_diff_gen/diff_celeb_x128_small_att.json # model 1.39M = 5.4MB -> bad
# submit_job celeb_diff_gen/diff_celeb_x128_deeper.json # model 162.24M = 620MB -> better
# submit_job celeb_diff_gen/diff_celeb_x128_more_layer.json # 2.12M = 8.2 MB -> bad
# submit_job celeb_diff_gen/diff_celeb_x128_wider.json # 18.46M = 71 MB -> better
# submit_job celeb_diff_gen/diff_celeb_x128_med.json # 66M = 250MB -> OK
# submit_job celeb_diff_gen/diff_celeb_x128.json # 3940.48M = 15GB -> OK


# submit_job celeb_diff_gen/diff_celeb_x128_seed_flat.json
# submit_job celeb_diff_gen/diff_celeb_x128_seed_narrow.json
# submit_job celeb_diff_gen/diff_celeb_x128_seed_noatt.json
# submit_job celeb_diff_gen/diff_celeb_x128_seed.json

# submit_job celeb_diff_gen/diff_celeb_x128_moreAtt.json
# submit_job celeb_diff_gen/diff_celeb_x256.json
# submit_job celeb_diff_gen/diff_celeb_x256_cos.json
# submit_job celeb_diff_gen/diff_celeb_x256_linsca.json
# submit_job celeb_diff_gen/diff_celeb_x256_sig.json

#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------
# submit_job celeb_vae/vae_celeb_x128.json
# submit_job celeb_vae/vae_celeb_x256.json
# submit_job celeb_vae/vae_celeb_x256_more_kl.json # -> this results in poor results (mu nicely centered at 0, but img quality bad)
# submit_job celeb_vae/vae_celeb_x256_less_kl.json


#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
# submit_job celeb_latent_diff/lat_diff_celeb_x128.json
# submit_job celeb_latent_diff/lat_diff_celeb_x256.json

submit_job celeb_Ldiff_gen_monai/dgL_m_cele_x256_v0.json

#--------------------------------------------------------------------------------------------------
# CBCT DIFF GEN
#--------------------------------------------------------------------------------------------------
# submit_job cbct_diff_gen/diff_cbct_x256.json # -> noisy
# submit_job cbct_diff_gen/diff_cbct_x256_minmax_norm.json
# submit_job cbct_diff_gen/diff_cbct_x256_sig1000.json
# submit_job cbct_diff_gen/diff_cbct_x256_attv1.json
# submit_job cbct_diff_gen/diff_cbct_x256_attv2.json
# submit_job cbct_diff_gen/diff_cbct_x256_01norm.json


# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v0.json
# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v1.json
# submit_job cbct_diff_gen_monai/dg_m_cbct_x256_v2.json


#--------------------------------------------------------------------------------------------------
# CBCT VAE
#--------------------------------------------------------------------------------------------------
# submit_job cbct_vae/vae_cbct_x256.json

#--------------------------------------------------------------------------------------------------
# CBCT LAT DIFF GEN
#--------------------------------------------------------------------------------------------------
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256.json
# submit_job cbct_latent_diff/lat_diff_cbct_maisi_x256_01norm.json