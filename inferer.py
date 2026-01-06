import argparse
import os
import sys
import warnings
import pickle
import time
from typing import Dict, Optional, Tuple

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

    candidates = []
    if model_path:
        candidates.append(os.path.abspath(os.path.expanduser(model_path)))
    else:
        candidates.append(os.path.abspath(os.path.join(root_dir, run_name)))

    for candidate in candidates:
        if os.path.isfile(candidate) and candidate.endswith(".ckpt"):
            return candidate, os.path.dirname(candidate)
        if os.path.isdir(candidate):
            resolved = _select_checkpoint_file(candidate)
            if resolved:
                return resolved, candidate

    raise FileNotFoundError(f"Unable to locate a checkpoint. Checked paths: {candidates}")


def load_model_from_checkpoint(
    checkpoint_path: str, config: Dict, device: torch.device
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the Lightning model checkpoint and return the underlying model + metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    model, metadata = load_model_from_checkpoint(checkpoint_path, config, device)

    solver_config = build_solver_config(config, args.num_inference_steps)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    ckpt_name = metadata["checkpoint_name"]
    output_path = os.path.join(
        checkpoint_dir,
        f"samples_{config_name}_{ckpt_name}_steps{solver_config['time_points']}.pkl",
    )
    print(f"Saving generated data to {output_path}")
    # if already exists, exit
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Exiting.")
        sys.exit()

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
    global_min, global_max = np.float32(np.inf), np.float32(-np.inf)

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
                global_min = np.minimum(global_min, image_np.min())
                global_max = np.maximum(global_max, image_np.max())

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

    # Save the generated data to the specified output path
    if global_max > global_min:
        offset = global_min
        scale = global_max - global_min
        for sample in generated_samples:
            sample["image"] = (sample["image"] - offset) / scale
    else:
        for sample in generated_samples:
            sample["image"] = np.zeros_like(sample["image"], dtype=np.float32)

    generated_dataset = {split_train_key: generated_samples}
    if split_val_key != split_train_key:
        generated_dataset[split_val_key] = generated_samples

    with open(output_path, "wb") as f:
        pickle.dump(generated_dataset, f)

    print(f"Generated data saved to {output_path}")


if __name__ == "__main__":
    main()
