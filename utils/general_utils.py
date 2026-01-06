import os
import yaml
import torch
import pickle
from typing import Dict, Optional

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
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_minusone_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [-1, 1].
    """
    return 2 * (tensor - tensor.min()) / (tensor.max() - tensor.min()) - 1


def load_and_prepare_data(
    pickle_path: str,
    split: str = "train",
    convert_classes_to_onehot: bool = False,
    is_ddpm: bool = False,
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

    # Vectorized assembly of tensors
    imgs = [torch.as_tensor(e["image"], dtype=torch.float32).squeeze(0) for e in data_split]
    mks = [torch.as_tensor(e["mask"], dtype=torch.float32).squeeze(0) for e in data_split]

    Images = torch.stack(imgs, dim=0).unsqueeze(1)  # [N, 1, H, W]
    Images = normalize_zero_to_one(Images)

    Masks = torch.stack(mks, dim=0).unsqueeze(1)  # [N, 1, H, W]
    Masks = normalize_zero_to_one(Masks)

    result = {"images": Images, "masks": Masks}

    has_class = "class" in data_split[0]
    if has_class:
        class_list = [e["class"] for e in data_split]
        if convert_classes_to_onehot:
            all_classes = sorted(set(class_list))
            class_to_idx = {c: i for i, c in enumerate(all_classes)}
            idx_to_class = {i: c for i, c in enumerate(all_classes)}
            idxs = torch.tensor([class_to_idx[c] for c in class_list], dtype=torch.long)
            onehot = torch.nn.functional.one_hot(idxs, num_classes=len(all_classes)).float()
            result["classes"] = onehot
            result["class_map"] = idx_to_class
        else:
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
