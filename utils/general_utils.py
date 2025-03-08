import os
import yaml
import torch
import pickle

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


def save_config(config, output_path: str):
    """
    Saves a config (Python dict) to a YAML file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    print(f"Config saved to: {output_path}")


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
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_minusone_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [-1, 1].
    """
    return 2 * (tensor - tensor.min()) / (tensor.max() - tensor.min()) - 1


def load_and_prepare_data(
    pickle_path: str,
    split: str = "train",
    new_masking: bool = False,
    threshold_mask: float = 0.15,
    convert_classes_to_onehot: bool = False,
    is_ddpm: bool = False,
):
    """
    Loads data from a pickle file containing a dict with:
      data_dict[split] -> list of dicts with keys ["image", "mask", "class", "name"]

    If 'use_masks_as_condition' is True, returns (X, Y=mask).
    Otherwise, returns (X, None).

    'new_masking' optionally modifies the mask using thresholding logic.

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

    image_list, mask_list, class_list = [], [], []
    for entry in data_split:
        # Load and normalize the image
        image = torch.tensor(entry["image"], dtype=torch.float32)  # [1, H, W]
        image_list.append(image)

        mask = torch.tensor(entry["mask"], dtype=torch.float32)  # [1, H, W]
        mask_list.append(mask)

        class_list.append(entry["class"])  # each is a string

    # Concatenate images and masks
    Images = torch.cat(image_list, dim=0)  # [N, H, W]
    assert Images.dim() == 3, f"Expected 3D tensor, got {Images.shape}"
    Images = Images.unsqueeze(1)  # [N, 1, H, W]
    Images = normalize_zero_to_one(Images)

    Masks = torch.cat(mask_list, dim=0)  # [N, H, W]
    assert Masks.dim() == 3, f"Expected 3D tensor, got {Masks.shape}"
    Masks = Masks.unsqueeze(1)  # [N, 1, H, W]

    if new_masking:
        # Apply new masking logic if required
        mask_new = torch.rand_like(Masks)
        for i in range(Images.shape[0]):
            mask_new[i] = (Images[i] > threshold_mask).float()
        # Combine mask_new with Y
        Masks = Masks * mask_new
        Masks = Masks + mask_new
    Masks = normalize_zero_to_one(Masks)

    # Convert classes to one-hot if required
    if convert_classes_to_onehot:
        all_classes = sorted(list(set(class_list)))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        idx_to_class = {i: c for i, c in enumerate(all_classes)}

        for i, c in enumerate(class_list):
            class_list[i] = class_to_idx[c]

        onehot_classes = torch.nn.functional.one_hot(
            torch.tensor(class_list), num_classes=len(all_classes)
        )  # [N, num_classes]
        Classes = onehot_classes.float()
        # if is_ddpm: # DDPM expects images in [-1, 1]
        if is_ddpm:
            Images = normalize_minusone_to_one(Images)

        return {"images": Images, "masks": Masks, "classes": Classes, "class_map": idx_to_class}
    else:
        Classes = class_list

    # if is_ddpm: # DDPM expects images in [-1, 1]
    if is_ddpm:
        Images = normalize_minusone_to_one(Images)

    return {"images": Images, "masks": Masks, "classes": Classes}


def create_dataloader(Images, Masks=None, classes=None, batch_size=8, shuffle=True):
    """
    Creates a DataLoader from Images, and optionally Masks and classes if given.
    """
    data_dict = {"images": Images}
    if Masks is not None:
        data_dict["masks"] = Masks
    if classes is not None:
        data_dict["classes"] = classes

    dataset = CustomDataset(data_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


###############################################################################
# Checkpointing
###############################################################################
def save_checkpoint(
    model,
    optimizer,
    epoch,
    config,
    checkpoint_dir: str,
    scheduler=None,
):
    """
    Saves model/optimizer states + config to "checkpoint.pth" inside 'checkpoint_dir'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved -> {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_dir,
    device,
    valid_only=False,
    scheduler=None,
):
    """
    Loads model/optimizer states + config from 'checkpoint.pth' in 'checkpoint_dir', if present.
    Returns (epoch, config).

    If 'valid_only' is True, optimizer state is not loaded (for pure inference).
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, None

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if not valid_only and optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and not valid_only:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    config_loaded = checkpoint.get("config", None)

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, config_loaded


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
