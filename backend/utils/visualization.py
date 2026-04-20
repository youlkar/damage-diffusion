# visualization for training monitoring and sample generation

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
import torchvision.utils as vutils


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1.0) / 2.0


def visualize_samples(
    images: torch.Tensor,
    masks: torch.Tensor,
    generated: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    # visualize image-mask pairs and optionally generated images
    num_samples = min(num_samples, images.shape[0])

    # determine grid size
    if generated is not None:
        num_rows = 3  # Real, Mask, Generated
        titles = ['Real Images', 'Masks', 'Generated Images']
    else:
        num_rows = 2  # Real, Mask
        titles = ['Real Images', 'Masks']

    fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 3, num_rows * 3))

    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    # move tensors to CPU and denormalize
    images_vis = denormalize(images[:num_samples]).cpu()
    masks_vis = masks[:num_samples].cpu()

    for i in range(num_samples):
        # real images
        img = images_vis[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(titles[0], fontsize=12, fontweight='bold')

        # masks
        mask = masks_vis[i].squeeze().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(titles[1], fontsize=12, fontweight='bold')

        # generated images (if provided)
        if generated is not None:
            gen_vis = denormalize(generated[:num_samples]).cpu()
            gen = gen_vis[i].permute(1, 2, 0).numpy()
            axes[2, i].imshow(gen)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title(titles[2], fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def save_samples(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
):
    # save a grid of images
    if normalize:
        images = denormalize(images)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, save_path, nrow=nrow)
    print(f"Saved image grid to {save_path}")


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    # plot training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig

def plot_metrics(
    metric: list,
    metric_label: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    # plot training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metric, label=metric_label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric} by Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig

def create_mask_overlay(
    images: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    # create overlay of masks on images for visualization
    images_vis = denormalize(images)

    # create red overlay for cracks
    overlay = images_vis.clone()
    overlay[:, 0, :, :] = torch.where(
        masks.squeeze(1) > 0.5,
        torch.ones_like(overlay[:, 0, :, :]),
        overlay[:, 0, :, :]
    )

    # blend
    result = alpha * overlay + (1 - alpha) * images_vis

    return result
