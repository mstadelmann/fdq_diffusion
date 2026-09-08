# Diffusion experiments using FDQ

Reference implementations of VAEs, pixel-space and latent diffusion, DiT/ViT
backbones, and flow matching, built on top of
[FDQ](https://github.com/mstadelmann/fonduecaquelon). Two domains: face
generation on CelebA-HQ and CT volume generation/reconstruction on CBCT/CT-RATE.

- https://github.com/mstadelmann/fdq_diffusion
- https://github.com/mstadelmann/fonduecaquelon
- https://github.com/CAIIVS/chuchichaestli

## Running an experiment

Locally:
```bash
pip install fdq[full]
git clone https://github.com/mstadelmann/fdq_diffusion.git
cd fdq_diffusion
fdq --config-path configs/celeb_vae --config-name vae_celeb_x128_00_base
```

On a Slurm cluster:

The `slurm_cluster` settings in these configs (partition, account, module
names, storage paths) are tuned for the
[ZHAW SoE Cluster](https://hpc.engineering.zhaw.ch/) (reachable only from
within the ZHAW network). Adjusting the partition/account names and the
system-specific paths will get this running on any Slurm cluster, or of
course locally wherever fdq can be installed.

```bash
# lightweight submit extra, no PyTorch required on the login node
pip install fdq[submit]
git clone https://github.com/mstadelmann/fdq_diffusion.git
cd fdq_diffusion
fdq_submit configs/celeb_vae/vae_celeb_x128_00_base.yaml
```

If package installation is restricted on the login node, use the standalone script:
```bash
wget https://raw.githubusercontent.com/mstadelmann/fonduecaquelon/refs/heads/main/src/fdq/submit.py
python submit.py configs/celeb_vae/vae_celeb_x128_00_base.yaml
```

`submit_celeb.sh` and `submit_ct.sh` list every configured experiment - run
them line by line, not as a script. Executing either file as-is submits every
experiment in it back to back, which is a lot of Slurm jobs and GPU-hours at
once.

## Experiments

| Directory | Contents |
|---|---|
| `configs/celeb_vae` | VAE pretraining, latent space for the models below |
| `configs/celeb_diff_gen_monai` | pixel-space DDPM, MONAI UNet |
| `configs/celeb_diff_gen_cc` | pixel-space DDPM, chuchichaestli UNet |
| `configs/celeb_diff_gen_vit` | pixel-space DDPM, DiT/ViT backbone |
| `configs/celeb_latent_diff` | latent DDPM, UNet backbone |
| `configs/celeb_latent_diff_vit` | latent DDPM, DiT/ViT backbone |
| `configs/celeb_fm_gen` | flow matching, UNet backbone |
| `configs/cbct_vae` | VAE pretraining on CT volumes |
| `configs/cbct_diff_gen`, `cbct_diff_gen_monai` | pixel-space DDPM on CT volumes |
| `configs/cbct_diff_cond_monai` | conditional DDPM, artifact/blur correction |
| `configs/cbct_latent_diff` | latent DDPM on CT volumes |

## Data

**Celeb**: [bitmind/celeb-a-hq](https://huggingface.co/datasets/bitmind/celeb-a-hq)
on Hugging Face - 30,000 images at native 1024x1024, CC-BY-4.0, no login or
token required. Downloaded once and cached locally; see `hf_preparator_celeb.py`.
Train/val/test are fixed, non-overlapping slices of that single split (by row
index, not random), so they're identical across runs.

**CBCT/CT**: currently trained on a locally preprocessed copy of
[CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) staged on
the cluster (see `nifity_preparator.py`, `data.ctrate.args.root_path`). Not
yet loaded directly from Hugging Face - migrating this to the same
download-once approach used for Celeb is planned.


## Results

### CELEB VAE

#### vae_celeb_x128_00_dist2.yaml
- GPU: 2x A100
- Train: 365 EP, 5h4m, 54s/EP

![vae_celeb_x128_00_dist2 results overview](assets/vae_celeb_x128_00_dist2.jpg)

#### vae_celeb_x256_00.yaml
- GPU: H100
- Train: 223 EP, 17h51m, 7m32s/EP

![vae_celeb_x256_00 results overview](assets/vae_celeb_x256_00.jpg)

#### vae_celeb_x512_00.yaml
- GPU: H200
- Train: 259 EP, 42h19m combined, 9m28s/EP (stopped early: ValLoss_stagnated)

![vae_celeb_x512_00 results overview](assets/vae_celeb_x512_00.jpg)

---

### CELEB DIFF GEN

#### MONAI UNet

##### dg_m_cele_x128_00_base_dist2.yaml
- GPU: 2x A100
- Train: 65 EP, 4h16m, 3m48s/EP

![dg_m_cele_x128_00_base_dist2 generated test samples](assets/dg_m_cele_x128_00_base_dist2.jpg)

##### dg_m_cele_x128_01.yaml (res-block off)
- GPU: 2x A100
- Train: 183 EP, 10h36m, 3m22s/EP

![dg_m_cele_x128_01 generated test samples](assets/dg_m_cele_x128_01.jpg)

##### dg_m_cele_x128_02.yaml (3 att levels)
- GPU: 2x A100
- Train: 153 EP, 19h58m, 7m42s/EP

![dg_m_cele_x128_02 generated test samples](assets/dg_m_cele_x128_02.jpg)

##### dg_m_cele_x128_03.yaml (4 layers only)
- GPU: 2x A100
- Train: 85 EP, 9h51m, 6m52s/EP

![dg_m_cele_x128_03 generated test samples](assets/dg_m_cele_x128_03.jpg)

##### dg_m_cele_x128_08_zero_snr.yaml (zero-SNR fix, v-pred)
- GPU: 2x A100
- Train: 96 EP, 6h16m, 3m46s/EP

![dg_m_cele_x128_08_zero_snr generated test samples](assets/dg_m_cele_x128_08_zero_snr.jpg)

##### dg_m_cele_x256_00.yaml (256px, 2 GPUs)
- GPU: 2x H200
- Train: 149 EP, 40h35m combined, 13m35s/EP (stopped early: TrainLoss_stagnated)

![dg_m_cele_x256_00 generated test samples](assets/dg_m_cele_x256_00.jpg)

##### dg_m_cele_x256_01_zero_snr.yaml (256px, zero-SNR fix)
- GPU: 2x H200
- Train: 109 EP, 11h2m, 6m0s/EP

![dg_m_cele_x256_01_zero_snr generated test samples](assets/dg_m_cele_x256_01_zero_snr.jpg)

#### Chuchichaestli UNet

##### dg_c_celeb_x128_00_base.yaml (baseline)
- GPU: 2x A100
- Train: 165 EP, 2h41m, 1m0s/EP

![dg_c_celeb_x128_00_base generated test samples](assets/dg_c_celeb_x128_00_base.jpg)

##### dg_c_celeb_x128_01.yaml (LR param, 4 parallel runs)
| Train | Image |
|---|---|
| 178 EP, 2h55m, 1m0s/EP | <img src="assets/dg_c_celeb_x128_01_p001.jpg" width="400"> |
| 116 EP, 1h52m, 54s/EP | <img src="assets/dg_c_celeb_x128_01_p002.jpg" width="400"> |
| 166 EP, 2h40m, 54s/EP | <img src="assets/dg_c_celeb_x128_01_p003.jpg" width="400"> |
| 146 EP, 2h22m, 59s/EP | <img src="assets/dg_c_celeb_x128_01_p004.jpg" width="400"> |

- GPU: 2x A100

##### dg_c_celeb_x128_02.yaml (lin/cos/sig param, 3 parallel runs)
| Train | Image |
|---|---|
| 200 EP, 3h11m, 56s/EP | <img src="assets/dg_c_celeb_x128_02_p001.jpg" width="400"> |
| 172 EP, 2h44m, 53s/EP | <img src="assets/dg_c_celeb_x128_02_p002.jpg" width="400"> |
| 216 EP, 3h31m, 54s/EP | <img src="assets/dg_c_celeb_x128_02_p003.jpg" width="400"> |

- GPU: 2x A100

##### dg_c_celeb_x256_00_base.yaml (256px)
- GPU: 2x H200
- Train: 130 EP, 2h29m, 1m5s/EP

![dg_c_celeb_x256_00_base generated test samples](assets/dg_c_celeb_x256_00_base.jpg)

#### VIT (DiT)

##### dg_vit_celeb_x128_00_base_dist2.yaml (2 GPUs)
- GPU: 2x A100
- Train: 350 EP, 56h33m combined, 9m32s/EP

![dg_vit_celeb_x128_00_base_dist2 generated test samples](assets/dg_vit_celeb_x128_00_base_dist2.jpg)

##### dg_vit_celeb_x256_00_base.yaml (256px, 2 GPUs)
- GPU: 2x H200
- Train: 302 EP, 35h16m combined, 10m50s/EP (stopped early: TrainLoss_stagnated)

![dg_vit_celeb_x256_00_base generated test samples](assets/dg_vit_celeb_x256_00_base.jpg)

---

### CELEB DIFF COND

#### dc_m_celeb_x128_deblur.yaml (MONAI UNet)
- GPU: A100
- Train: 126 EP, 16h9m, 7m41s/EP (stopped early: ValLoss_stagnated)

![dc_m_celeb_x128_deblur generated test samples](assets/dc_m_celeb_x128_deblur.jpg)

#### dc_cc_celeb_x128_deblur.yaml (chuchichaestli UNet)
- GPU: 2x A100
- Train: 154 EP, 2h49m, 1m6s/EP (stopped early: ValLoss_stagnated)

![dc_cc_celeb_x128_deblur generated test samples](assets/dc_cc_celeb_x128_deblur.jpg)

---

### CELEB Latent Diffusion

#### lat_diff_celeb_x128_00_base.yaml (UNet backbone)
- GPU: H200
- Train: 224 EP, 15h2m, 4m2s/EP (stopped early: ValLoss_stagnated)

![lat_diff_celeb_x128_00_base generated test samples](assets/lat_diff_celeb_x128_00_base.jpg)

#### lat_diff_celeb_x256_00.yaml (256px)
- GPU: H200
- Train: 174 EP, 13h5m, 4m31s/EP (stopped early: TrainLoss_stagnated)

![lat_diff_celeb_x256_00 generated test samples](assets/lat_diff_celeb_x256_00.jpg)

#### VIT (DiT) backbone, same pretrained VAEs

##### lat_diff_vit_celeb_x128_00_base.yaml
- GPU: H200
- Train: 400 EP, 13h27m, 2m2s/EP

![lat_diff_vit_celeb_x128_00_base generated test samples](assets/lat_diff_vit_celeb_x128_00_base.jpg)

##### lat_diff_vit_celeb_x256_00.yaml (256px)
- GPU: H200
- Train: 400 EP, 52h29m combined, 7m47s/EP

![lat_diff_vit_celeb_x256_00 generated test samples](assets/lat_diff_vit_celeb_x256_00.jpg)

### CELEB Flow Matching Generative

#### fm_meta_cele_x128_v0.yaml (baseline)

- GPU: 2x H200
- Train: 650 EP, 84h20m combined, 7m47s/EP

![fm_meta_cele_x128_v0 generated test samples](assets/fm_meta_cele_x128_v0.png)

#### fm_meta_cele_x128_v1.yaml (smaller batch)
- GPU: 2x H200

![fm_meta_cele_x128_v1 generated test samples](assets/fm_meta_cele_x128_v1.png)

#### fm_meta_cele_x128_v2.yaml (EMA, attn tweaks)
- GPU: 2x H200


#### fm_meta_cele_x256_v0.yaml (256px)
- GPU: 2x H200
