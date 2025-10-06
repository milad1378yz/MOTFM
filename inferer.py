import argparse
import os
import sys
import warnings
import pickle
import time
from typing import Dict, Optional, Tuple

from tqdm import tqdm

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch

# Ensure the script can find the utils modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.general_utils import create_dataloader, load_and_prepare_data, load_config
from utils.utils_fm import sample_batch

from trainer import FlowMatchingLightningModule


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
    data_config = config["data_args"]
    val_data = load_and_prepare_data(
        pickle_path=data_config["pickle_path"],
        split=data_config["split_val"],
        new_masking=True,
        convert_classes_to_onehot=True,
    )
    val_loader = create_dataloader(
        Images=val_data["images"],
        Masks=val_data["masks"],
        classes=val_data.get("classes"),
        batch_size=config["train_args"]["batch_size"],
        shuffle=False,
    )

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
    # Initialize containers for generated data
    generated_data = {"train": []}  # Assuming training data is not generated here

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

            # Assuming batch is a dictionary with keys 'images', 'masks', 'classes'
            images = batch["images"]
            masks = batch.get("masks") if mask_conditioning else None
            classes = batch.get("classes") if class_conditioning else None

            # Move tensors to CPU and convert to appropriate formats
            final_imgs = final_imgs.cpu().detach().numpy()  # shape: (B, C, H, W)
            masks = masks.cpu().detach().numpy() if masks is not None else None
            classes = (
                classes.cpu().detach().numpy() if classes is not None else None
            )  # one-hot encoded (B, num_classes)
            batch_size = final_imgs.shape[0]  # B

            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break

                sample_dict = {
                    "image": final_imgs[i],  # Assuming final_imgs are already normalized
                    "mask": masks[i] if masks is not None else None,
                    "class": (
                        idx_to_class[classes[i].argmax()]
                        if (idx_to_class is not None and classes is not None)
                        else None
                    ),
                    "name": f"sample_{samples_collected}",
                    "true_data": images[i],
                }

                generated_data["train"].append(sample_dict)
                samples_collected += 1
                pbar.update(1)

    print(f"Collected {samples_collected} samples.")

    # Calculate and print average time per sample
    average_time_per_sample = total_time / samples_collected
    print(f"Average time per sample: {average_time_per_sample:.4f} seconds")

    # Save the generated data to the specified output path
    with open(output_path, "wb") as f:
        pickle.dump(generated_data, f)

    print(f"Generated data saved to {output_path}")


if __name__ == "__main__":
    main()
