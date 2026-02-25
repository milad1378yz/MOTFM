# 3D Evaluation

This folder contains a simple script to compute 3D metrics between a generated dataset and a reference dataset.

## Metrics

The script computes:

- `MMD` (MONAI `compute_mmd`)
- `MS-SSIM` (MONAI `compute_ms_ssim`)
- `FID` for 3D volumes using `torchvision.models.video.r3d_18` features + MONAI `FIDMetric`

## Usage

```bashس
python evaluation_3d/evaluate_3d.py \
  --generated_path /path/to/generated.pkl \
  --reference_path /path/to/reference.pkl \
  --generated_split train \
  --reference_split valid \
  --num_samples 200
```

Input expectations:
- Both inputs are `.pkl` files with list-like splits (`train`, `valid`, etc.).
- Generated samples should contain `image`.
- Reference samples should contain `image` (or `true_data`).
- Volumes are expected as `[C, D, H, W]` or `[D, H, W]`.

Notes:

- Use `--skip_fid` to skip 3D FID if `torchvision` video weights are unavailable.
- Normalization options before metric computation:
  - `per_set_minmax` (default)
  - `shared_minmax`
  - `none`
