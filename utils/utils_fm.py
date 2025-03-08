import os
import json
import torch
import matplotlib.pyplot as plt
from torch import nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver

from .general_utils import normalize_zero_to_one, save_image
from tqdm import tqdm


###############################################################################
# Model Building
###############################################################################
class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """

    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep

        # If controlnet is None, we won't do anything special in forward.
        self.has_controlnet = controlnet is not None
        self.has_conditioning = unet.with_conditioning

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        """
        Args:
            x: input image tensor [B, C, H, W].
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: [B,1 , conditions_dim].
            masks: [B, C, H, W] masks for conditioning.

        Returns:
            The network output (e.g. velocity, noise, or predicted epsilon).
        """
        # Scale continuous t -> discrete timesteps(If you dont want to change the embedding function in the UNet)
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # t's shape should be [B]

        if self.has_controlnet:
            # cond is expected to be a ControlNet conditioning, e.g. mask
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            # If no ControlNet, cond might be cross-attention or None
            output = self.unet(x=x, timesteps=t, context=cond)

        return output


def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """
    Builds a model (UNet only, or UNet+ControlNet) based on the provided model_config.

    Args:
        model_config: Dictionary containing model configuration.
        device: Device to move the model to.

    Returns:
        A MergedModel instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a copy so the original config remains unaltered.
    mc = model_config.copy()

    # Pop out keys that are not needed by the model constructors.
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    # Pop out ControlNet specific key, if present.
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build the base UNet by passing all remaining items as kwargs.
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        # Ensure the controlnet has its specific key.
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
        # Pass the same config kwargs to ControlNet plus the controlnet-specific key.
        controlnet = ControlNet(**mc, conditioning_embedding_num_channels=cond_embed_channels)
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    # Print number of trainable parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    model_size_mb = num_params * 4 / (1024**2)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model.to(device)


def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    solver = ODESolver(velocity_model=model)

    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=True,
        cond=cond,
        masks=masks,
    )
    return sol


def plot_solver_steps(sol, im_batch, mask_batch, class_batch, class_map, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    if mask_batch is not None:
        fig, axes = plt.subplots(n_samples, n_steps + 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    for i in range(n_samples):
        for t in range(n_steps):
            axes[i][t].imshow(sol[t, i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            axes[i][col].imshow(mask_batch[i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        axes[i][col].imshow(im_batch[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
        if class_map and class_batch is not None:
            idx = class_batch[i].argmax().item()
            cls = class_map[idx] if idx < len(class_map) else str(idx)
            axes[i][col].text(
                0.5,
                -0.15,
                f"Class: {cls}",
                ha="center",
                va="top",
                transform=axes[i][col].transAxes,
                color="red",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


def validate_and_save_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=16,
    class_map=None,
    mask_conditioning=True,
    class_conditioning=False,
):
    model.eval()
    outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    count, step_plot_done = 0, False
    for batch in tqdm(val_loader, desc="Validating"):
        imgs = batch["images"].to(device)
        cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
        masks = batch["masks"].to(device) if mask_conditioning else None

        x_init = torch.randn_like(imgs)
        sol = sample_with_solver(
            model,
            x_init,
            solver_config,
            cond=cond,
            masks=masks,
        )
        final_imgs = sol[-1] if sol.dim() == 5 else sol
        for i in range(final_imgs.size(0)):
            if count >= max_samples:
                break
            gen_img = normalize_zero_to_one(final_imgs[i])
            real_img = normalize_zero_to_one(imgs[i])
            sdir = os.path.join(outdir, f"sample_{count+1:03d}")
            os.makedirs(sdir, exist_ok=True)
            save_image(gen_img, os.path.join(sdir, "gen.png"))
            save_image(real_img, os.path.join(sdir, "real.png"))
            if masks is not None:
                cnd_img = normalize_zero_to_one(masks[i])
                save_image(cnd_img, os.path.join(sdir, "mask.png"))
            if class_map and "classes" in batch:
                idx = batch["classes"][i].argmax().item()
                with open(os.path.join(sdir, "class.json"), "w") as f:
                    json.dump(
                        {
                            "class_index": idx,
                            "class_name": class_map[idx] if idx < len(class_map) else str(idx),
                            "class_map": class_map,
                            "class_coditioning": class_conditioning,
                            "mask_conditioning": mask_conditioning,
                        },
                        f,
                        indent=4,
                    )
            count += 1
        if not step_plot_done:
            clz = batch["classes"] if class_map and "classes" in batch else None
            plot_solver_steps(sol, imgs, masks, clz, class_map, outdir)
            step_plot_done = True
        if count >= max_samples:
            break
    print(f"Validation samples saved in: {outdir}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = False,
):
    model.eval()
    imgs = batch["images"].to(device)
    cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
    masks = batch["masks"].to(device) if mask_conditioning else None

    x_init = torch.randn_like(imgs)
    sol = sample_with_solver(
        model=model, solver_config=solver_config, x_init=x_init, cond=cond, masks=masks
    )
    final_imgs = sol[-1] if sol.dim() == 5 else sol
    return final_imgs
