#!/bin/bash

submit_job() {
    root_path="/cluster/home/stmd/dev/fdq_diffusion/configs/"
    python3 /cluster/home/stmd/dev/fonduecaquelon/fdq_submit.py $root_path$1
}

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
#--------------------------------------------------------------------------------------------------

# submit_job celeb_diff_gen/diff_celeb_x128.json
submit_job celeb_diff_gen/diff_celeb_x128_moreAtt.json
# submit_job celeb_diff_gen/diff_celeb_x256.json
# submit_job celeb_diff_gen/diff_celeb_x256_cos.json
# submit_job celeb_diff_gen/diff_celeb_x256_linsca.json
# submit_job celeb_diff_gen/diff_celeb_x256_sig.json

#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------

submit_job celeb_vae/vae_celeb_x128.json
submit_job celeb_vae/vae_celeb_x256.json
# submit_job celeb_vae/vae_celeb_x256_more_kl.json # -> this results in poor results (mu nicely centered at 0, but img quality bad)
# submit_job celeb_vae/vae_celeb_x256_less_kl.json

#--------------------------------------------------------------------------------------------------
# CELEB Latent Diffusion
#--------------------------------------------------------------------------------------------------
# submit_job celeb_latent_diff/lat_diff_celeb_x128.json
submit_job celeb_latent_diff/lat_diff_celeb_x256.json