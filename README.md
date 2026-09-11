# Material-Guided Decoders: Joint Estimation of Material Properties and HDRI Lighting

**Single-image HDR lighting estimation, conditioned on predicted material parameters.**
PyTorch + Blender. Columbia University, Deep Learning for Computer Graphics, Spring 2026.

**[Full write-up (PDF)](https://www.shanyingliu.com/docs/material-guided-decoders-writeup.pdf)**

![Teaser](teaser.png)

## Summary

Inverse rendering treats lighting and material as entangled unknowns, but most lighting estimators never use material information explicitly. This project asks whether predicting an object's material and feeding that prediction back into the lighting decoder produces more physically plausible environment maps than a direct image-to-envmap regression.

Two models predict a `64×128` equirectangular HDR environment map from a single `128×128` render of a sphere:

- **Baseline** — ImageNet-pretrained ResNet-50 encoder -> transposed-conv decoder -> HDR envmap.
- **Material-guided (multi-task)** — same encoder, plus a head regressing five Principled BSDF parameters (metallic, roughness, specular, transmission, IOR). The predicted material drives the decoder via FiLM (per-channel `(1+γ)·x+β` modulation in every upsample block), with a tiled-concat variant as an ablation.

Training uses a log-HDR loss (peak-weighted MSE + L1 + SSIM) so sun pixels, which carry the structure, are not drowned out by dark sky. Evaluation goes beyond pixel metrics: predicted envmaps are used as light probes to **re-render a Utah teapot in Cycles**, and the renders are scored with LPIPS against ground truth.

**Result:** the baseline wins on pixel-wise log-MSE, but the material-guided model produces noticeably more convincing relighting, especially on glossy/metallic objects (roughness < 0.2, metallic > 0.8), where the surface reflection acts as a specular anchor for reconstructing the environment. The gap between the two rankings is itself a finding: pixel-wise envmap metrics are a poor proxy for relighting quality.

![Baseline vs. material-guided relighting](envcomp.png)

## Pipeline

```
Blender (data_gen.py)         : sphere renders + metadata.json  (5 material classes, ~15k samples)
preprocess_hdris.py           : HDRIs cached as 64×128 float32 .npy
main.py --mode both           : train baseline + multitask, evaluate, plot
  └─ utils/teapot_compare.py  : re-render teapots under predicted envmaps, LPIPS
```

```bash
cd code
python main.py --mode both               # train baseline + multitask, then evaluate
python main.py --mode both --eval-only   # evaluate saved checkpoints
```

**Note about branches:** `main` holds the baseline and the tiled-concat multi-task model. The FiLM-conditioned `MaterialGuidedLightingNet` (the "material-guided" variant in the write-up) and the seed-sweep tooling live on the [`sweep`](../../tree/sweep) branch (`--multitask-arch film`).

Datasets, checkpoints and render caches are not tracked (see `.gitignore`); everything is regenerable from the scripts above. Checkpoints available on request.

## Layout

| Path | What |
| --- | --- |
| `code/models/` | `BaselineLightingNet`, `MaterialAwareLightingNet` (tiled), `MaterialGuidedLightingNet` (FiLM), decoders |
| `code/training/` | `Trainer`, log-HDR loss terms, from-scratch SSIM |
| `code/evaluation/` | log-MSE, LPIPS, per-material / per-attribute slicing, paired t-tests |
| `code/data_gen.py`, `code/render_teapot.py` | Blender scripts for dataset generation and teapot relighting |
| `docs/Overview.md` | Detailed architecture, loss, and evaluation notes |
| `result/material_aware_lighting/` | Evaluation figures |

## Future work

Generalizing to real, multi-object scenes via segmentation; a spherical mover's loss to address the spatial misalignment seen in envmap-space metrics; splitting prediction into luminance/chromaticity and diffuse/specular components to improve hue consistency and reflection handling.
