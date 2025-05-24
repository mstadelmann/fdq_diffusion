# Diffusion experiments using FDQ

- https://github.com/mstadelmann/fdq_diffusion
- https://github.com/mstadelmann/fonduecaquelon
- https://github.com/CAIIVS/chuchichaestli

To start an experiment, run e.g.
```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
git clone https://github.com/mstadelmann/fdq_diffusion.git
cd fonduecaquelon
pip install .
fdq <path_to_fdq_diffusion>configs/celeb_vae/celeb_vae_x128.json
```

To submit an experiment to the SoE Slurm cluster, run e.g.
```bash
git clone https://github.com/mstadelmann/fonduecaquelon.git
git clone https://github.com/mstadelmann/fdq_diffusion.git
python <path_to_fdq>/fdq_submit.py <path_to_fdq_diffusion>configs/celeb_vae/celeb_vae_x128.json
```