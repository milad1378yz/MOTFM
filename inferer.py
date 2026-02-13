import argparse
import os
import sys
import warnings
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import numpy as np

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch

# Ensure the script can find the utils modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.general_utils import load_config
from utils.utils_fm import sample_batch

from trainer import FlowMatchingDataModule, FlowMatchingLightningModule


def _select_checkpoint_file(ckpt_dir: str) -> Optional[str]:
    """
    Pick an appropriate checkpoint file from a directory.
    Preference order: last.ckpt -> most recent .ckpt file.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    candidates = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _dedupe_preserve_order(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def resolve_checkpoint_path(
    model_path: Optional[str], config: Dict, config_path: str
) -> Tuple[str, str]:
    """
    Resolve a user-provided checkpoint path or infer it from config/run name.

    Returns:
        checkpoint_path (str): absolute path to the checkpoint file.
        checkpoint_dir (str): directory containing the checkpoint.
    """
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    root_dir = config["train_args"]["checkpoint_dir"]

    candidates: List[str] = []
    if model_path:
        base = os.path.abspath(os.path.expanduser(model_path))
        candidates.extend(
            [
                base,
                os.path.join(base, run_name),
                os.path.join(base, "latest"),
            ]
        )
    else:
        candidates.append(os.path.abspath(os.path.join(root_dir, run_name)))
    candidates = _dedupe_preserve_order(candidates)

    for candidate in candidates:
        if os.path.isfile(candidate) and candidate.endswith(".ckpt"):
            return candidate, os.path.dirname(candidate)
        if os.path.isdir(candidate):
            resolved = _select_checkpoint_file(candidate)
            if resolved:
                return resolved, candidate

    raise FileNotFoundError(f"Unable to locate a checkpoint. Checked paths: {candidates}")


def _get_nested(cfg: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, bool]:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _extract_checkpoint_config(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hp = checkpoint.get("hyper_parameters")
    if not isinstance(hp, dict):
        return None
    if "model_args" in hp and "train_args" in hp:
        return hp
    nested = hp.get("config")
    if isinstance(nested, dict):
        return nested
    return None


def validate_checkpoint_config(
    checkpoint: Dict[str, Any], config: Dict[str, Any], allow_mismatch: bool = False
) -> None:
    """
    Compare critical model fields between the current config and the config
    saved inside the checkpoint to avoid silently loading the wrong run.
    """
    ckpt_config = _extract_checkpoint_config(checkpoint)
    if ckpt_config is None:
        print(
            "Warning: checkpoint has no recoverable saved config; skipping compatibility checks."
        )
        return

    critical_fields = [
        ("model_args", "spatial_dims"),
        ("model_args", "in_channels"),
        ("model_args", "out_channels"),
        ("model_args", "num_channels"),
        ("model_args", "num_res_blocks"),
        ("model_args", "attention_levels"),
        ("model_args", "transformer_num_layers"),
        ("model_args", "with_conditioning"),
        ("model_args", "mask_conditioning"),
        ("model_args", "cross_attention_dim"),
        ("model_args", "conditioning_embedding_num_channels"),
    ]

    mismatches = []
    for path in critical_fields:
        ckpt_val, has_ckpt = _get_nested(ckpt_config, path)
        cfg_val, has_cfg = _get_nested(config, path)
        if has_ckpt and has_cfg and ckpt_val != cfg_val:
            mismatches.append((path, ckpt_val, cfg_val))

    if not mismatches:
        return

    lines = ["Checkpoint/config mismatch detected in critical fields:"]
    for path, ckpt_val, cfg_val in mismatches:
        lines.append(f"  - {'.'.join(path)}: checkpoint={ckpt_val!r}, current_config={cfg_val!r}")
    lines.append("Use `--allow_config_mismatch` only if this is intentional.")
    message = "\n".join(lines)

    if allow_mismatch:
        print("Warning:", message)
    else:
        raise ValueError(message)


def _normalize_sample_image(img: np.ndarray, mode: str) -> np.ndarray:
    x = np.nan_to_num(img.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return x
    if mode == "clip_0_1":
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    if mode == "per_sample_minmax":
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max > x_min:
            return ((x - x_min) / (x_max - x_min)).astype(np.float32, copy=False)
        return np.zeros_like(x, dtype=np.float32)
    raise ValueError(f"Unsupported output normalization mode: {mode}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    allow_config_mismatch: bool = False,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the Lightning model checkpoint and return the underlying model + metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint_config(
        checkpoint=checkpoint,
        config=config,
        allow_mismatch=allow_config_mismatch,
    )
    lightning_module = FlowMatchingLightningModule(config)
    lightning_module.load_state_dict(checkpoint["state_dict"], strict=True)
    model = lightning_module.model.to(device)
    model.eval()
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "checkpoint_name": os.path.splitext(os.path.basename(checkpoint_path))[0],
    }
    return model, metadata


def build_solver_config(config: Dict, num_inference_steps: Optional[int]) -> Dict:
    solver_config = dict(config.get("solver_args", {}))
    solver_config.setdefault("method", "midpoint")
    if num_inference_steps:
        solver_config["time_points"] = num_inference_steps
        solver_config["step_size"] = 1.0 / num_inference_steps
    solver_config.setdefault("time_points", 10)
    solver_config.setdefault("step_size", 1.0 / solver_config["time_points"])
    return solver_config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Inference script for the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to save. If None, all samples are saved.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint file or directory. If omitted, "
            "`train_args.checkpoint_dir/<config_basename>` is used."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
        help="Number of inference steps during sampling.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Output .pkl path. If omitted, a name derived from config/checkpoint/steps is used in "
            "the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output_path if it already exists.",
    )
    parser.add_argument(
        "--output_norm",
        type=str,
        default="clip_0_1",
        choices=["clip_0_1", "per_sample_minmax", "global_minmax", "none"],
        help=(
            "Normalization applied to generated images before saving. "
            "`clip_0_1` avoids global contrast collapse."
        ),
    )
    parser.add_argument(
        "--allow_config_mismatch",
        action="store_true",
        help="Allow loading a checkpoint whose saved config mismatches the current config.",
    )

    args = parser.parse_args()

    # Load config and determine checkpoint path
    config_path = args.config_path

    config = load_config(config_path)
    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(args.model_path, config, config_path)
    print(f"Using checkpoint: {checkpoint_path}")

    # Device setup
    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Inference device:", device)

    # Build model and load checkpoint
    model, metadata = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    print(
        f"Checkpoint metadata: epoch={metadata['epoch']}, "
        f"global_step={metadata['global_step']}"
    )

    solver_config = build_solver_config(config, args.num_inference_steps)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    ckpt_name = metadata["checkpoint_name"]
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(
            checkpoint_dir,
            f"samples_{config_name}_{ckpt_name}_steps{solver_config['time_points']}.pkl",
        )
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        if args.overwrite:
            print(f"Overwriting existing output file: {output_path}")
        else:
            base, ext = os.path.splitext(output_path)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            ext = ext or ".pkl"
            output_path = f"{base}_{timestamp}{ext}"
            print(
                "Output file already exists and --overwrite was not set. "
                f"Writing to: {output_path}"
            )

    print(f"Saving generated data to {output_path}")

    # Load dataset for inference
    datamodule = FlowMatchingDataModule(config)
    datamodule.setup(stage="validate")
    val_data = datamodule.val_data
    if val_data is None:
        raise RuntimeError("Failed to initialize validation data for inference.")
    val_loader = datamodule.val_dataloader()

    # Determine the number of samples to infer
    dataset_size = len(val_loader.dataset)
    num_samples = dataset_size if args.num_samples is None else args.num_samples
    print(f"Number of samples to save: {num_samples}")

    print(
        f"Solver config: method={solver_config['method']}, "
        f"step_size={solver_config['step_size']}, "
        f"time_points={solver_config['time_points']}"
    )

    idx_to_class = val_data.get("class_map")
    model_args = config.get("model_args", {})
    class_conditioning = bool(model_args.get("with_conditioning", False))
    mask_conditioning = bool(model_args.get("mask_conditioning", False))
    # Save using the same split keys expected by the trainer config.
    data_args = config.get("data_args", {})
    split_train_key = data_args.get("split_train", "train")
    split_val_key = data_args.get("split_val", "valid")
    generated_samples = []
    raw_global_min, raw_global_max = np.float32(np.inf), np.float32(-np.inf)

    samples_collected = 0
    total_time = 0  # Initialize total time

    # Create an iterator that can be reset when the dataset is exhausted
    val_iterator = iter(val_loader)

    with tqdm(total=num_samples, desc="Generating Samples") as pbar:
        while samples_collected < num_samples:
            try:
                batch = next(val_iterator)
            except StopIteration:
                # Reinitialize the iterator if the dataset is exhausted
                val_iterator = iter(val_loader)
                batch = next(val_iterator)

            start_time = time.time()  # Start time for the batch

            # Sample batch
            final_imgs = sample_batch(
                model=model,
                solver_config=solver_config,
                batch=batch,
                device=device,
                class_conditioning=class_conditioning,
                mask_conditioning=mask_conditioning,
            )

            end_time = time.time()  # End time for the batch
            batch_time = end_time - start_time
            total_time += batch_time  # Accumulate total time

            if mask_conditioning and "masks" not in batch:
                raise KeyError(
                    "mask_conditioning is enabled but the dataloader batch has no 'masks' key."
                )

            masks_t = batch.get("masks")
            classes_t = batch.get("classes")

            # Move tensors to CPU and convert to numpy.
            final_imgs = final_imgs.detach().cpu().numpy()  # shape: (B, C, ...)
            masks_np = masks_t.detach().cpu().numpy() if masks_t is not None else None
            classes_np = classes_t.detach().cpu().numpy() if classes_t is not None else None
            batch_size = final_imgs.shape[0]  # B

            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break

                image_np = final_imgs[i].astype(np.float32, copy=False)
                raw_global_min = np.minimum(raw_global_min, image_np.min())
                raw_global_max = np.maximum(raw_global_max, image_np.max())

                class_value = (
                    idx_to_class[int(classes_np[i].argmax())]
                    if (idx_to_class is not None and classes_np is not None)
                    else None
                )

                sample_dict = {
                    "image": image_np,
                    "mask": (
                        masks_np[i].astype(np.float32, copy=False) if masks_np is not None else None
                    ),
                    "name": f"sample_{samples_collected}",
                }
                if class_value is not None:
                    sample_dict["class"] = class_value

                generated_samples.append(sample_dict)
                samples_collected += 1
                pbar.update(1)

    print(f"Collected {samples_collected} samples.")

    # Calculate and print average time per sample
    average_time_per_sample = total_time / samples_collected
    print(f"Average time per sample: {average_time_per_sample:.4f} seconds")

    print(f"Raw generated range: [{float(raw_global_min):.6f}, {float(raw_global_max):.6f}]")
    print(f"Applying output normalization mode: {args.output_norm}")

    if args.output_norm == "global_minmax":
        if raw_global_max > raw_global_min:
            offset = raw_global_min
            scale = raw_global_max - raw_global_min
            for sample in generated_samples:
                sample["image"] = ((sample["image"] - offset) / scale).astype(
                    np.float32, copy=False
                )
        else:
            for sample in generated_samples:
                sample["image"] = np.zeros_like(sample["image"], dtype=np.float32)
    else:
        for sample in generated_samples:
            sample["image"] = _normalize_sample_image(sample["image"], args.output_norm)

    generated_dataset = {split_train_key: generated_samples}
    if split_val_key != split_train_key:
        generated_dataset[split_val_key] = generated_samples

    with open(output_path, "wb") as f:
        pickle.dump(generated_dataset, f)

    print(f"Generated data saved to {output_path}")


if __name__ == "__main__":
    main()
