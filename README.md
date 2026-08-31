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
- slurm train: 49397
- slurm test: 49615
- GPU: 2x A100
- Train: 365 EP, 5h4m, 54s/EP

![vae_celeb_x128_00_dist2 results overview](assets/vae_celeb_x128_00_dist2.jpg)

#### vae_celeb_x256_00.yaml
- slurm train: 49398
- slurm test: 50014
- GPU: H100
- Train: 223 EP, 17h51m, 7m32s/EP

![vae_celeb_x256_00 results overview](assets/vae_celeb_x256_00.jpg)

#### vae_celeb_x512_00.yaml
- slurm train: 49399
- slurm test: none (training failed)
- GPU: H200
- Error: out of memory - job killed by Slurm (exit 137)

---

### CELEB DIFF GEN

#### MONAI UNet

##### dg_m_cele_x128_00_base_dist2.yaml
- slurm train: 49401
- slurm test: 49616
- GPU: 2x A100
- Train: 65 EP, 4h16m, 3m48s/EP

![dg_m_cele_x128_00_base_dist2 generated test samples](assets/dg_m_cele_x128_00_base_dist2.jpg)

##### dg_m_cele_x128_01.yaml (res-block off)
- slurm train: 49402
- slurm test: 49796
- GPU: 2x A100
- Train: 183 EP, 10h36m, 3m22s/EP

![dg_m_cele_x128_01 generated test samples](assets/dg_m_cele_x128_01.jpg)

##### dg_m_cele_x128_02.yaml (3 att levels)
- slurm train: 49403
- slurm test: 49998
- GPU: 2x A100
- Train: 153 EP, 19h58m, 7m42s/EP

![dg_m_cele_x128_02 generated test samples](assets/dg_m_cele_x128_02.jpg)

##### dg_m_cele_x128_03.yaml (4 layers only)
- slurm train: 49404
- slurm test: 49868
- GPU: 2x A100
- Train: 85 EP, 9h51m, 6m52s/EP

![dg_m_cele_x128_03 generated test samples](assets/dg_m_cele_x128_03.jpg)

##### dg_m_cele_x128_08_zero_snr.yaml (zero-SNR fix, v-pred)
- slurm train: 49405
- slurm test: 49942
- GPU: 2x A100
- Train: 96 EP, 6h16m, 3m46s/EP

![dg_m_cele_x128_08_zero_snr generated test samples](assets/dg_m_cele_x128_08_zero_snr.jpg)

##### dg_m_cele_x256_00.yaml (256px, 2 GPUs)
- slurm train: 49406 → 49950 (auto-resubmitted after 49406 hit the walltime limit)
- slurm test: 50548
- GPU: 2x H200
- Train: 149 EP, 40h35m combined, 13m35s/EP (stopped early: TrainLoss_stagnated)

![dg_m_cele_x256_00 generated test samples](assets/dg_m_cele_x256_00.jpg)

##### dg_m_cele_x256_01_zero_snr.yaml (256px, zero-SNR fix)
- slurm train: 49407
- slurm test: 49835
- GPU: 2x H200
- Train: 109 EP, 11h2m, 6m0s/EP

![dg_m_cele_x256_01_zero_snr generated test samples](assets/dg_m_cele_x256_01_zero_snr.jpg)

#### Chuchichaestli UNet

##### dg_c_celeb_x128_00_base.yaml (baseline)
- slurm train: 49408
- slurm test: 49943
- GPU: 2x A100
- Train: 165 EP, 2h41m, 1m0s/EP

![dg_c_celeb_x128_00_base generated test samples](assets/dg_c_celeb_x128_00_base.jpg)

##### dg_c_celeb_x128_01.yaml (LR param, 4 parallel runs)
| slurm train | slurm test | Train | Image |
|---|---|---|---|
| 49409 | 49967 | 178 EP, 2h55m, 1m0s/EP | <img src="assets/dg_c_celeb_x128_01_p001.jpg" width="200"> |
| 49410 | 49948 | 116 EP, 1h52m, 54s/EP | <img src="assets/dg_c_celeb_x128_01_p002.jpg" width="200"> |
| 49411 | 49993 | 166 EP, 2h40m, 54s/EP | <img src="assets/dg_c_celeb_x128_01_p003.jpg" width="200"> |
| 49412 | 49995 | 146 EP, 2h22m, 59s/EP | <img src="assets/dg_c_celeb_x128_01_p004.jpg" width="200"> |

- GPU: 2x A100

##### dg_c_celeb_x128_02.yaml (lin/cos/sig param, 3 parallel runs)
| slurm train | slurm test | Train | Image |
|---|---|---|---|
| 49413 | 50000 | 200 EP, 3h11m, 56s/EP | <img src="assets/dg_c_celeb_x128_02_p001.jpg" width="200"> |
| 49414 | 50001 | 172 EP, 2h44m, 53s/EP | <img src="assets/dg_c_celeb_x128_02_p002.jpg" width="200"> |
| 49415 | 50003 | 216 EP, 3h31m, 54s/EP | <img src="assets/dg_c_celeb_x128_02_p003.jpg" width="200"> |

- GPU: 2x A100

##### dg_c_celeb_x256_00_base.yaml (256px)
- slurm train: 49416
- slurm test: 49866
- GPU: 2x H200
- Train: 130 EP, 2h29m, 1m5s/EP

![dg_c_celeb_x256_00_base generated test samples](assets/dg_c_celeb_x256_00_base.jpg)

#### VIT (DiT)

##### dg_vit_celeb_x128_00_base_dist2.yaml (2 GPUs)
- slurm train: 49418 → 50410 → 50531 (auto-resubmitted twice after hitting the walltime limit)
- slurm test: 50591
- GPU: 2x A100
- Train: 350 EP, 56h33m combined, 9m32s/EP

![dg_vit_celeb_x128_00_base_dist2 generated test samples](assets/dg_vit_celeb_x128_00_base_dist2.jpg)

##### dg_vit_celeb_x256_00_base.yaml (256px, 2 GPUs)
- slurm train: 49419 → 50265 (auto-resubmitted after 49419 hit the walltime limit)
- slurm test: 50547
- GPU: 2x H200
- Train: 302 EP, 35h16m combined, 10m50s/EP (stopped early: TrainLoss_stagnated)

![dg_vit_celeb_x256_00_base generated test samples](assets/dg_vit_celeb_x256_00_base.jpg)

---

### CELEB DIFF COND

#### dc_m_celeb_x128_deblur.yaml (MONAI UNet)
- slurm train: 49420
- slurm test: none (training failed)
- GPU: A100
- Error: crashed in epoch 1 - RuntimeError, tensor size mismatch concatenating the conditioning input during sampling

#### dc_cc_celeb_x128_deblur.yaml (chuchichaestli UNet)
- slurm train: 49421
- slurm test: 50004
- GPU: 2x A100
- Train: 128 EP, 2h10m, 55s/EP

![dc_cc_celeb_x128_deblur generated test samples](assets/dc_cc_celeb_x128_deblur.jpg)

---

### CELEB Latent Diffusion

#### lat_diff_celeb_x128_00_base.yaml (UNet backbone)
- slurm train: 49422
- slurm test: none (training failed)
- GPU: H200
- Error: crashed in epoch 1 - KeyError: 'MSE' (loss key missing from config)

#### lat_diff_celeb_x256_00.yaml (256px)
- slurm train: 49423
- slurm test: none (training failed)
- GPU: H200
- Error: crashed in epoch 1 - KeyError: 'MSE' (loss key missing from config)

#### VIT (DiT) backbone, same pretrained VAEs

##### lat_diff_vit_celeb_x128_00_base.yaml
- slurm train: 49424
- slurm test: 50002
- GPU: H200
- Train: 400 EP, 13h27m, 2m2s/EP

![lat_diff_vit_celeb_x128_00_base generated test samples](assets/lat_diff_vit_celeb_x128_00_base.jpg)

##### lat_diff_vit_celeb_x256_00.yaml (256px)
- slurm train: 49425 → 50286 → 50596 (auto-resubmitted twice after hitting the walltime limit)
- slurm test: 50698
- GPU: H200
- Train: 400 EP, 52h29m combined, 7m47s/EP

![lat_diff_vit_celeb_x256_00 generated test samples](assets/lat_diff_vit_celeb_x256_00.jpg)

### CELEB Flow Matching Generative

#### fm_meta_cele_x128_v0.yaml (baseline)
- slurm train: 49426
- slurm test: none (training failed)
- GPU: 2x H200
- Error: crashed before completing epoch 1 - DDP worker process aborted (SIGABRT) during startup

#### fm_meta_cele_x128_v1.yaml (smaller batch)
- slurm train: 49427 → 50341
- slurm test: none (training failed)
- GPU: 2x H200
- Train (49427, before crash): ~408 EP, 23h3m, 3m20s/EP
- Error: 49427 hit the walltime limit and auto-resubmitted; 50341 crashed immediately on resume - DDP worker process aborted (SIGABRT)

#### fm_meta_cele_x128_v2.yaml (EMA, attn tweaks)
- slurm train: 49428 → 50371
- slurm test: none (training failed)
- GPU: 2x H200
- Train (49428, before crash): ~130 EP, 22h59m, 9m54s/EP
- Error: 49428 hit the walltime limit and auto-resubmitted; 50371 crashed immediately on resume - DDP worker process aborted (SIGABRT)

#### fm_meta_cele_x256_v0.yaml (256px)
- slurm train: 49429 → 50447 → 50684
- slurm test: none (training failed)
- GPU: 2x H200
- Train (49429 + 50447, before crash): ~16 EP over ~42h, ~160m/EP
- Error: 49429 and 50447 each hit the walltime limit and auto-resubmitted; 50684 crashed immediately on resume - DDP worker process aborted (SIGABRT)