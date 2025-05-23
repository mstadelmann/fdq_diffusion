#!/bin/bash

submit_job() {
    root_path="/cluster/home/stmd/dev/fdq_diffusion/configs/"
    python3 /cluster/home/stmd/dev/fonduecaquelon/fdq_submit.py $root_path$1
}

#--------------------------------------------------------------------------------------------------
# CELEB DIFF GEN
#--------------------------------------------------------------------------------------------------

submit_job diff_celeb_gen/generate_faces_x128.json
submit_job diff_celeb_gen/generate_faces_x128_moreAtt.json
submit_job diff_celeb_gen/generate_faces_x256.json
submit_job diff_celeb_gen/generate_faces_x256_cos.json
submit_job diff_celeb_gen/generate_faces_x256_linsca.json
submit_job diff_celeb_gen/generate_faces_x256_sig.json

#--------------------------------------------------------------------------------------------------
# CELEB VAE
#--------------------------------------------------------------------------------------------------

submit_job celeb_vae/celeb_vae_x128.json
submit_job celeb_vae/celeb_vae_x256.json
submit_job celeb_vae/celeb_vae_x256_more_kl.json
submit_job celeb_vae/celeb_vae_x256_no_kl.json
