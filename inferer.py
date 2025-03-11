import argparse
import os
import sys
import warnings
import pickle
from tqdm import tqdm
import time

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch

# Ensure the script can find the utils modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.general_utils import (
    load_config,
    load_checkpoint,
    load_and_prepare_data,
    create_dataloader,
)
from utils.utils_fm import build_model, sample_batch


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
        default="mask_class_conditioning_checkpoints/latest",
        help="Path to the model checkpoint to load.",
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
    model_path = (
        os.path.join(config["train_args"]["checkpoint_dir"], "latest")
        if args.model_path is None
        else args.model_path
    )

    # Device setup
    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Inference device:", device)

    # Build model and load checkpoint
    model = build_model(config["model_args"], device=device)
    start_epoch, _ = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_dir=model_path,
        device=device,
        valid_only=True,
    )

    output_path = (
        f"fm_epoch_{start_epoch}_num_inference_{args.num_inference_steps}_"
        + config_path.split("/")[-1].replace(".yaml", ".pkl")
    )
    output_path = os.path.join(config["train_args"]["checkpoint_dir"], output_path)
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
        classes=val_data["classes"],
        batch_size=config["train_args"]["batch_size"],
        shuffle=False,
    )

    # Determine the number of samples to infer
    dataset_size = len(val_loader.dataset)
    num_samples = dataset_size if args.num_samples is None else args.num_samples
    print(f"Number of samples to save: {num_samples}")

    step_size = 1.0 / args.num_inference_steps
    print(f"Step size: {step_size}")

    # Sampling parameters
    solver_config = {
        "method": config["train_args"].get("method", "midpoint"),
        "step_size": step_size,
        "time_points": config["train_args"].get("time_points", 10),
    }

    idx_to_class = val_data["class_map"]
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
                class_conditioning=config["general_args"]["class_conditioning"],
                mask_conditioning=config["general_args"]["mask_conditioning"],
            )

            end_time = time.time()  # End time for the batch
            batch_time = end_time - start_time
            total_time += batch_time  # Accumulate total time

            # Assuming batch is a dictionary with keys 'images', 'masks', 'classes'
            images = batch["images"]
            masks = batch["masks"]
            classes = batch["classes"]

            # Move tensors to CPU and convert to appropriate formats
            final_imgs = final_imgs.cpu().detach().numpy()  # shape: (B, C, H, W)
            masks = masks.cpu().detach().numpy()  # same shape as final_imgs
            classes = classes.cpu().detach().numpy()  # one-hot encoded (B, num_classes)
            batch_size = final_imgs.shape[0]  # B

            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break

                sample_dict = {
                    "image": final_imgs[i],  # Assuming final_imgs are already normalized
                    "mask": masks[i] if masks is not None else None,
                    "class": idx_to_class[classes[i].argmax()],
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
