import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 3D metrics between a generated dataset and a reference dataset."
    )
    parser.add_argument(
        "--generated_path",
        type=str,
        required=True,
        help="Path to generated dataset pickle.",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference dataset pickle.",
    )
    parser.add_argument(
        "--generated_split",
        type=str,
        default="train",
        help="Split key in generated dataset (default: train).",
    )
    parser.add_argument(
        "--reference_split",
        type=str,
        default="valid",
        help="Split key in reference dataset (default: valid).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: min(len(generated), len(reference))).",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["per_set_minmax", "shared_minmax", "none"],
        default="per_set_minmax",
        help="Normalization before metric computation (default: per_set_minmax).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for metric computation (default: auto).",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=8,
        help="Batch size for R3D-18 feature extraction used by 3D FID.",
    )
    parser.add_argument(
        "--ms_ssim_weights",
        type=str,
        default="0.1,0.3,0.6",
        help="Comma-separated weights for MS-SSIM levels (default: 0.1,0.3,0.6).",
    )
    parser.add_argument(
        "--ms_ssim_kernel_size",
        type=int,
        default=5,
        help="Kernel size for MS-SSIM (default: 5).",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip 3D FID computation.",
    )
    return parser.parse_args()


def _parse_weights(weights_csv: str) -> Tuple[float, ...]:
    try:
        weights = tuple(float(x.strip()) for x in weights_csv.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --ms_ssim_weights='{weights_csv}'.") from exc
    if not weights:
        raise ValueError("MS-SSIM weights cannot be empty.")
    return weights


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def _load_pickle(path: Path) -> Dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(data).__name__}.")
    return data


def _resolve_split(data: Dict, split_name: str, label: str, path: Path) -> List[dict]:
    if split_name not in data:
        available = [k for k, v in data.items() if isinstance(v, list)]
        raise ValueError(f"{label} split '{split_name}' not found in {path}. Available: {available}")
    split = data[split_name]
    if not isinstance(split, list):
        raise ValueError(f"{label} split '{split_name}' must be a list, got {type(split).__name__}.")
    if len(split) == 0:
        raise ValueError(f"{label} split '{split_name}' is empty.")
    return split


def _ensure_channel_first_3d(tensor: torch.Tensor, *, source: str) -> torch.Tensor:
    x = torch.as_tensor(tensor).squeeze()
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"{source}: expected 3D/4D volume, got shape {tuple(x.shape)}.")

    if x.shape[0] <= 4:
        return x
    if x.shape[-1] <= 4:
        return x.permute(3, 0, 1, 2).contiguous()
    return x


def _extract_volume(entry: dict, *, source: str) -> torch.Tensor:
    if "image" in entry:
        return _ensure_channel_first_3d(
            torch.as_tensor(entry["image"], dtype=torch.float32), source=f"{source}['image']"
        )
    if "true_data" in entry:
        return _ensure_channel_first_3d(
            torch.as_tensor(entry["true_data"], dtype=torch.float32), source=f"{source}['true_data']"
        )
    raise KeyError(f"{source}: expected one of keys ['image', 'true_data'].")


def _pair_volumes(
    generated_entries: Sequence[dict],
    reference_entries: Sequence[dict],
    num_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_pairs = min(len(generated_entries), len(reference_entries))
    if max_pairs <= 0:
        raise ValueError("No overlapping samples between generated and reference splits.")

    n = max_pairs if num_samples is None else min(int(num_samples), max_pairs)
    if n <= 0:
        raise ValueError(f"Invalid requested sample count: {num_samples}.")

    true_volumes: List[torch.Tensor] = []
    generated_volumes: List[torch.Tensor] = []

    for i in range(n):
        gen_vol = _extract_volume(generated_entries[i], source=f"generated[{i}]")
        ref_vol = _extract_volume(reference_entries[i], source=f"reference[{i}]")
        if gen_vol.shape != ref_vol.shape:
            raise ValueError(
                f"Shape mismatch at sample {i}: generated {tuple(gen_vol.shape)} "
                f"vs reference {tuple(ref_vol.shape)}."
            )
        generated_volumes.append(gen_vol)
        true_volumes.append(ref_vol)

    true_stack = torch.stack(true_volumes, dim=0)
    gen_stack = torch.stack(generated_volumes, dim=0)
    return true_stack, gen_stack


def _minmax(tensor: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    denom = (x_max - x_min).clamp_min(1e-8)
    return (x - x_min) / denom


def _normalize_pair(
    true_images: torch.Tensor,
    generated_images: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "none":
        return (
            torch.nan_to_num(true_images.float(), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(generated_images.float(), nan=0.0, posinf=0.0, neginf=0.0),
        )

    if mode == "per_set_minmax":
        return _minmax(true_images), _minmax(generated_images)

    if mode == "shared_minmax":
        true_images = torch.nan_to_num(true_images.float(), nan=0.0, posinf=0.0, neginf=0.0)
        generated_images = torch.nan_to_num(
            generated_images.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_min = torch.minimum(torch.amin(true_images), torch.amin(generated_images))
        x_max = torch.maximum(torch.amax(true_images), torch.amax(generated_images))
        denom = (x_max - x_min).clamp_min(1e-8)
        return (true_images - x_min) / denom, (generated_images - x_min) / denom

    raise ValueError(f"Unsupported normalization mode '{mode}'.")


def _to_three_channels(volumes: torch.Tensor) -> torch.Tensor:
    c = int(volumes.shape[1])
    if c == 3:
        return volumes
    if c < 3:
        repeats = (3 + c - 1) // c
        return volumes.repeat(1, repeats, 1, 1, 1)[:, :3]
    return volumes[:, :3]


def _load_r3d18_backbone(device: torch.device) -> torch.nn.Module:
    try:
        import torchvision
    except Exception as exc:
        raise ImportError(
            "torchvision is required for 3D FID. Install torchvision or use --skip_fid."
        ) from exc

    try:
        model = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained r3d_18 weights for 3D FID. Use --skip_fid."
        ) from exc

    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _extract_r3d_features(
    model: torch.nn.Module,
    volumes: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for i in range(0, int(volumes.shape[0]), int(batch_size)):
        batch = volumes[i : i + batch_size].to(device)
        feats.append(model(batch).detach().cpu())
    return torch.cat(feats, dim=0)


def compute_metrics(
    true_images: torch.Tensor,
    generated_images: torch.Tensor,
    *,
    ms_ssim_weights: Tuple[float, ...],
    ms_ssim_kernel_size: int,
    device: torch.device,
    fid_batch_size: int,
    skip_fid: bool,
) -> Dict[str, Union[float, None]]:
    try:
        from monai.metrics import FIDMetric, compute_mmd, compute_ms_ssim
    except Exception as exc:
        raise ImportError(
            "MONAI metrics are required. Install monai (or ensure monai_generative installed MONAI)."
        ) from exc

    true_images = true_images.to(device)
    generated_images = generated_images.to(device)

    mmd_value = compute_mmd(true_images, generated_images, None)
    try:
        ms_ssim_value = compute_ms_ssim(
            true_images,
            generated_images,
            3,
            weights=ms_ssim_weights,
            kernel_size=int(ms_ssim_kernel_size),
        )
    except ValueError as exc:
        raise ValueError(
            "MS-SSIM failed. If volumes are small, use fewer --ms_ssim_weights levels "
            "or smaller --ms_ssim_kernel_size."
        ) from exc
    metrics: Dict[str, Union[float, None]] = {
        "mmd": float(mmd_value.item()),
        "ms_ssim": float(ms_ssim_value.mean().item()),
        "fid": None,
    }

    if skip_fid:
        return metrics

    fid_metric = FIDMetric()
    r3d_model = _load_r3d18_backbone(device)
    real_features = _extract_r3d_features(
        r3d_model, _to_three_channels(true_images), device, fid_batch_size
    )
    gen_features = _extract_r3d_features(
        r3d_model, _to_three_channels(generated_images), device, fid_batch_size
    )
    fid_score = fid_metric(real_features, gen_features)
    metrics["fid"] = float(fid_score.item())
    return metrics


def main() -> None:
    args = parse_args()
    weights = _parse_weights(args.ms_ssim_weights)
    device = _resolve_device(args.device)

    generated_path = Path(args.generated_path).expanduser().resolve()
    reference_path = Path(args.reference_path).expanduser().resolve()
    if not generated_path.exists():
        raise FileNotFoundError(f"--generated_path does not exist: {generated_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"--reference_path does not exist: {reference_path}")

    generated_data = _load_pickle(generated_path)
    reference_data = _load_pickle(reference_path)
    generated_entries = _resolve_split(
        generated_data, args.generated_split, label="Generated", path=generated_path
    )
    reference_entries = _resolve_split(
        reference_data, args.reference_split, label="Reference", path=reference_path
    )

    true_images, generated_images = _pair_volumes(
        generated_entries, reference_entries, args.num_samples
    )
    true_images, generated_images = _normalize_pair(true_images, generated_images, args.normalization)

    metrics = compute_metrics(
        true_images,
        generated_images,
        ms_ssim_weights=weights,
        ms_ssim_kernel_size=args.ms_ssim_kernel_size,
        device=device,
        fid_batch_size=args.fid_batch_size,
        skip_fid=args.skip_fid,
    )

    print(f"Generated: {generated_path} [{args.generated_split}]")
    print(f"Reference: {reference_path} [{args.reference_split}]")
    print(f"Samples used: {int(generated_images.shape[0])}")
    print(f"MMD: {metrics['mmd']:.6f}")
    print(f"MS-SSIM: {metrics['ms_ssim']:.6f}")
    if metrics["fid"] is None:
        print("FID: skipped")
    else:
        print(f"FID: {metrics['fid']:.6f}")


if __name__ == "__main__":
    main()
