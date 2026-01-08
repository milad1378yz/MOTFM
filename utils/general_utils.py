import os
import yaml
import torch
import pickle
import warnings
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset


###############################################################################
# Config Handling
###############################################################################
def load_config(config_path: str = "config.yaml"):
    """
    Loads a YAML config file from the given path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# Data Loading & Preparation
###############################################################################
class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data_dict.items()}


def normalize_zero_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [0, 1].
    """
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    t_min = tensor.min()
    t_max = tensor.max()
    denom = (t_max - t_min).clamp_min(1e-6)
    return (tensor - t_min) / denom


def normalize_minusone_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [-1, 1].
    """
    return 2 * normalize_zero_to_one(tensor) - 1


def _normalize_minmax(
    tensor: torch.Tensor,
    out_range: Tuple[float, float] = (0.0, 1.0),
    *,
    scope: str = "global",
    eps: float = 1e-6,
    clip_percentiles: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Min-max normalize with guards for constant tensors and non-finite values."""
    x = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if scope not in {"global", "sample", "sample_channel"}:
        raise ValueError(f"Invalid scope='{scope}'. Use 'global', 'sample', or 'sample_channel'.")

    reduce_dims = None
    if scope == "sample":
        reduce_dims = tuple(range(1, x.ndim))
    elif scope == "sample_channel":
        reduce_dims = tuple(range(2, x.ndim))

    if clip_percentiles is not None:
        lo, hi = clip_percentiles
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(
                f"clip_percentiles must satisfy 0 <= lo < hi <= 100, got {clip_percentiles}."
            )
        q_lo = torch.quantile(x, lo / 100.0, dim=reduce_dims, keepdim=reduce_dims is not None)
        q_hi = torch.quantile(x, hi / 100.0, dim=reduce_dims, keepdim=reduce_dims is not None)
        x = x.clamp(q_lo, q_hi)

    if reduce_dims is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = torch.amin(x, dim=reduce_dims, keepdim=True)
        x_max = torch.amax(x, dim=reduce_dims, keepdim=True)

    denom = (x_max - x_min).clamp_min(eps)
    y = (x - x_min) / denom

    out_min, out_max = out_range
    return y * (out_max - out_min) + out_min


def _normalize_zscore(
    tensor: torch.Tensor,
    *,
    scope: str = "global",
    eps: float = 1e-6,
    clip_percentiles: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Z-score normalize with guards for constant tensors and non-finite values."""
    x = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if scope not in {"global", "sample", "sample_channel"}:
        raise ValueError(f"Invalid scope='{scope}'. Use 'global', 'sample', or 'sample_channel'.")

    reduce_dims = None
    if scope == "sample":
        reduce_dims = tuple(range(1, x.ndim))
    elif scope == "sample_channel":
        reduce_dims = tuple(range(2, x.ndim))

    if clip_percentiles is not None:
        lo, hi = clip_percentiles
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(
                f"clip_percentiles must satisfy 0 <= lo < hi <= 100, got {clip_percentiles}."
            )
        q_lo = torch.quantile(x, lo / 100.0, dim=reduce_dims, keepdim=reduce_dims is not None)
        q_hi = torch.quantile(x, hi / 100.0, dim=reduce_dims, keepdim=reduce_dims is not None)
        x = x.clamp(q_lo, q_hi)

    if reduce_dims is None:
        mean = x.mean()
        std = x.std(unbiased=False)
    else:
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, unbiased=False, keepdim=True)

    return (x - mean) / std.clamp_min(eps)


def apply_normalization(
    tensor: torch.Tensor,
    *,
    mode: str = "minmax",
    scope: str = "global",
    clip_percentiles: Optional[Tuple[float, float]] = None,
    out_range: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply a normalization strategy with sane defaults and guards.

    Supported modes:
      - "none": no scaling (non-finite values -> 0)
      - "minmax": min-max scaling to out_range
      - "minmax_0_1": alias for minmax with (0, 1)
      - "minmax_-1_1": alias for minmax with (-1, 1)
      - "zscore": zero-mean/unit-variance
      - "auto": keep if already in out_range (approx), else minmax
    """
    mode = (mode or "minmax").lower()
    scope = (scope or "global").lower()

    x = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if mode in {"none", "identity", "raw"}:
        return x

    if mode in {"minmax_0_1", "0_1"}:
        return _normalize_minmax(
            x, out_range=(0.0, 1.0), scope=scope, eps=eps, clip_percentiles=clip_percentiles
        )

    if mode in {"minmax_-1_1", "-1_1", "m1_1"}:
        return _normalize_minmax(
            x, out_range=(-1.0, 1.0), scope=scope, eps=eps, clip_percentiles=clip_percentiles
        )

    if mode == "minmax":
        return _normalize_minmax(
            x, out_range=out_range, scope=scope, eps=eps, clip_percentiles=clip_percentiles
        )

    if mode == "zscore":
        return _normalize_zscore(x, scope=scope, eps=eps, clip_percentiles=clip_percentiles)

    if mode == "auto":
        try:
            x_min = float(x.min())
            x_max = float(x.max())
        except Exception:
            return _normalize_minmax(
                x, out_range=out_range, scope=scope, eps=eps, clip_percentiles=clip_percentiles
            )

        out_min, out_max = out_range
        # Fast path: already approximately in range.
        if x_min >= out_min - 1e-3 and x_max <= out_max + 1e-3:
            return x

        # Common case: input in [-1, 1] but we want [0, 1], or vice-versa.
        if out_range == (0.0, 1.0) and x_min >= -1.0 - 1e-3 and x_max <= 1.0 + 1e-3:
            return (x + 1.0) / 2.0
        if out_range == (-1.0, 1.0) and x_min >= 0.0 - 1e-3 and x_max <= 1.0 + 1e-3:
            return x * 2.0 - 1.0

        return _normalize_minmax(
            x, out_range=out_range, scope=scope, eps=eps, clip_percentiles=clip_percentiles
        )

    raise ValueError(
        f"Unknown normalization mode '{mode}'. "
        "Use one of: none, minmax, minmax_0_1, minmax_-1_1, zscore, auto."
    )


def load_and_prepare_data(
    pickle_path: str,
    split: str = "train",
    convert_classes_to_onehot: bool = False,
    is_ddpm: bool = False,
    *,
    image_norm: Union[str, Dict] = "minmax_0_1",
    mask_norm: Union[str, Dict] = "minmax_0_1",
    spatial_dims: Optional[int] = None,
    norm_scope: str = "global",
    clip_percentiles: Optional[Tuple[float, float]] = None,
    norm_eps: float = 1e-6,
    class_to_idx: Optional[Dict[object, int]] = None,
    num_classes: Optional[int] = None,
    class_mapping_split: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Loads data from a pickle file containing a dict with:
      data_dict[split] -> list of dicts with keys ["image", "mask", "class", "name"]

    If 'use_masks_as_condition' is True, returns (X, Y=mask).
    Otherwise, returns (X, None).

    Returns:
      X: [N, C, H, W] float tensor
      Y or None: [N, C, H, W], if use_masks_as_condition is True
      (H, W): dimensions
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    # Extract data for the specified split
    data_split = data_dict.get(split, [])
    if not data_split:
        raise ValueError(f"No data found for split '{split}' in the pickle file.")

    def _ensure_channel_first(x: torch.Tensor, *, name: str) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"Expected '{name}' to have at least 2 dims, got shape {tuple(x.shape)}"
            )

        if spatial_dims is not None:
            if x.ndim == spatial_dims:
                return x.unsqueeze(0)
            if x.ndim == spatial_dims + 1:
                return x
            raise ValueError(
                f"Unexpected '{name}' shape {tuple(x.shape)} for spatial_dims={spatial_dims}. "
                f"Expected {spatial_dims}D (no channel) or {spatial_dims+1}D (channel-first)."
            )

        # Heuristic fallback: assume channel-first if the leading dim is small.
        if x.ndim == 2:
            return x.unsqueeze(0)
        if x.ndim == 3 and x.shape[0] > 8:
            return x.unsqueeze(0)
        return x

    # Assemble tensors while preserving channels.
    imgs = [_ensure_channel_first(torch.as_tensor(e["image"], dtype=torch.float32), name="image") for e in data_split]
    Images = torch.stack(imgs, dim=0)  # [N, C, ...]
    if isinstance(image_norm, dict):
        image_mode = str(image_norm.get("mode", "minmax_0_1"))
        image_scope = str(image_norm.get("scope", norm_scope))
        image_clip = image_norm.get("clip_percentiles", clip_percentiles)
        image_range = tuple(image_norm.get("range", (0.0, 1.0)))
        image_eps = float(image_norm.get("eps", norm_eps))
    else:
        image_mode = str(image_norm)
        image_scope = norm_scope
        image_clip = clip_percentiles
        image_range = (0.0, 1.0)
        image_eps = norm_eps

    Images = apply_normalization(
        Images,
        mode=image_mode,
        scope=image_scope,
        clip_percentiles=image_clip,
        out_range=image_range,
        eps=image_eps,
    )

    result = {"images": Images}

    has_mask = any("mask" in e for e in data_split)
    if has_mask:
        if not all("mask" in e for e in data_split):
            raise ValueError(f"Split '{split}' has inconsistent samples: some are missing the 'mask' key.")
        mks = [_ensure_channel_first(torch.as_tensor(e["mask"], dtype=torch.float32), name="mask") for e in data_split]
        Masks = torch.stack(mks, dim=0)  # [N, C, ...]

        if isinstance(mask_norm, dict):
            mask_mode = str(mask_norm.get("mode", "minmax_0_1"))
            mask_scope = str(mask_norm.get("scope", norm_scope))
            mask_clip = mask_norm.get("clip_percentiles", clip_percentiles)
            mask_range = tuple(mask_norm.get("range", (0.0, 1.0)))
            mask_eps = float(mask_norm.get("eps", norm_eps))
        else:
            mask_mode = str(mask_norm)
            mask_scope = norm_scope
            mask_clip = clip_percentiles
            mask_range = (0.0, 1.0)
            mask_eps = norm_eps

        Masks = apply_normalization(
            Masks,
            mode=mask_mode,
            scope=mask_scope,
            clip_percentiles=mask_clip,
            out_range=mask_range,
            eps=mask_eps,
        )
        result["masks"] = Masks
    else:
        # Keep downstream code stable: only warn if masks are expected/used elsewhere.
        warnings.warn(f"Split '{split}' has no 'mask' key; returning images only.")

    has_class = any("class" in e for e in data_split)
    if has_class:
        if not all("class" in e for e in data_split):
            raise ValueError(
                f"Split '{split}' has inconsistent samples: some are missing the 'class' key."
            )
        class_list = [e["class"] for e in data_split]
        if convert_classes_to_onehot:
            if class_to_idx is None:
                mapping_source = class_mapping_split or split
                mapping_split = data_dict.get(mapping_source, [])
                if not mapping_split:
                    raise ValueError(
                        f"class_mapping_split='{mapping_source}' is empty or missing in the pickle."
                    )
                if not any("class" in e for e in mapping_split):
                    raise ValueError(
                        f"class_mapping_split='{mapping_source}' has no 'class' key in its samples."
                    )
                all_classes = sorted({e["class"] for e in mapping_split if "class" in e})
                if not all_classes:
                    raise ValueError(
                        f"class_mapping_split='{mapping_source}' has no usable class values."
                    )
                class_to_idx = {c: i for i, c in enumerate(all_classes)}

            if num_classes is None:
                num_classes = len(class_to_idx)
            if int(num_classes) != len(class_to_idx):
                raise ValueError(
                    f"num_classes={num_classes} does not match discovered classes "
                    f"({len(class_to_idx)}). Set `data_args.class_values` (preferred) or "
                    f"update `model_args.cross_attention_dim` to match."
                )

            unknown = sorted({c for c in class_list if c not in class_to_idx})
            if unknown:
                raise ValueError(
                    f"Found class values not present in the class mapping: {unknown}. "
                    f"Consider defining `data_args.class_values` to fix the mapping/order."
                )

            idxs = torch.tensor([class_to_idx[c] for c in class_list], dtype=torch.long)
            onehot = torch.nn.functional.one_hot(idxs, num_classes=int(num_classes)).float()
            result["classes"] = onehot
            result["class_map"] = {i: c for c, i in class_to_idx.items()}
        else:
            # Keep as tensor for consistent batching when classes are numeric.
            try:
                result["classes"] = torch.as_tensor(class_list)
            except Exception:
                result["classes"] = class_list

    if is_ddpm:
        result["images"] = normalize_minusone_to_one(result["images"])

    return result


def create_dataloader(
    Images: torch.Tensor,
    Masks: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    sampler=None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
):
    """
    Creates a performant DataLoader from tensors, optionally including masks/classes.

    Additional knobs:
      - num_workers: dataloader workers
      - pin_memory: defaults to True on CUDA machines
      - persistent_workers: defaults to True if num_workers > 0 and not shuffle-only epoch
    """
    data_dict = {"images": Images}
    if Masks is not None:
        data_dict["masks"] = Masks
    if classes is not None:
        data_dict["classes"] = classes

    dataset = CustomDataset(data_dict)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if sampler is not None and shuffle:
        warnings.warn("Both sampler and shuffle were set; disabling shuffle in favor of sampler.")
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


###############################################################################
# Image Saving
###############################################################################
def save_image(img_tensor, out_path):
    """
    Saves a single 2D image (assumed shape [1, H, W] or [H, W]) as PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # remove batch/channel dims if present
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    plt.figure()
    plt.imshow(img_tensor.cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_image_3d(img_tensor, out_path, slice_idx=None):
    """
    Saves a single 3D image (assumed shape [1, D, H, W] or [D, H, W]) as a series of PNGs.
    If slice_idx is provided, saves only that slice.
    """
    os.makedirs(out_path, exist_ok=True)
    # remove batch/channel dims if present
    if img_tensor.dim() == 4 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() != 3:
        raise ValueError("img_tensor must be 3D (D, H, W)")

    D = img_tensor.shape[0]
    slice_indices = [slice_idx] if slice_idx is not None else [D // 2]

    for i in slice_indices:
        plt.figure()
        plt.imshow(img_tensor[i].cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(out_path, f"slice_{i:03d}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()
