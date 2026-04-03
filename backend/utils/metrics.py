# Evaluation metrics for generative model quality assessment.
# Includes FID (Fréchet Inception Distance), IoU, and pixel accuracy calculations.

import torch
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchvision import transforms


def _build_inception_model(device: str) -> InceptionV3:
    """Initialize InceptionV3 once and reuse — avoids reloading every FID call."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def _preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] → [0, 1], resize to 299x299, normalize for InceptionV3."""
    images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
    resize = transforms.Resize((299, 299))
    images = torch.stack([resize(img) for img in images])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    images = torch.stack([normalize(img) for img in images])
    return images


def _extract_features(
    images: torch.Tensor,
    inception_model: InceptionV3,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract InceptionV3 features from images in batches."""
    images_proc = _preprocess_for_inception(images).to(device)
    features = []
    with torch.no_grad():
        for i in range(0, len(images_proc), batch_size):
            batch = images_proc[i:i + batch_size]
            feat = inception_model(batch)[0].squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = 'cuda',
    inception_model: Optional[InceptionV3] = None,  # FIX: pass in to avoid reloading
) -> float:
    """
    Compute FID between real and generated images.

    Args:
        real_images: shape (N, 3, H, W), values in [-1, 1]
        generated_images: shape (N, 3, H, W), values in [-1, 1]
        device: 'cuda' or 'cpu'
        inception_model: pre-initialized InceptionV3 (optional, avoids reload each call)
    Returns:
        FID score (lower is better)
    """
    if inception_model is None:
        inception_model = _build_inception_model(device)

    real_features = _extract_features(real_images, inception_model, device)
    gen_features = _extract_features(generated_images, inception_model, device)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union (IoU) for binary masks.

    Args:
        pred_mask: Predicted mask, shape (B, 1, H, W) or (1, H, W)
        true_mask: Ground truth mask, shape (B, 1, H, W) or (1, H, W)
        threshold: Threshold for binarization
    Returns:
        Mean IoU score across batch
    """
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()
    intersection = (pred_binary * true_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + true_binary.sum(dim=(-2, -1)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_pixel_accuracy(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()
    correct = (pred_binary == true_binary).float().sum()
    total = pred_binary.numel()
    return (correct / total).item()


def evaluate_conditioning_alignment(
    model,
    val_loader,
    num_samples: int = 100,
    device: str = 'cuda',
) -> Tuple[float, float]:
    # TODO: Implement with pre-trained segmentation model
    # For now, return placeholder values
    print("Warning: Conditioning alignment evaluation requires a pre-trained segmentation model.")
    print("This will be implemented in Phase 4 (Evaluation).")
    return 0.0, 0.0


class MetricsTracker:
    """Track and log metrics during training."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # FIX: added iou and pixel_accuracy tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'fid_score': [],
            'iou': [],              # added
            'pixel_accuracy': [],   # added
            'learning_rate': [],
            'epoch': [],
        }
        # FIX: initialize InceptionV3 once for reuse across FID calls
        self._inception_model = None
        self._inception_device = None

    def get_inception_model(self, device: str) -> InceptionV3:
        """Lazy-load InceptionV3 once and reuse."""
        if self._inception_model is None or self._inception_device != device:
            self._inception_model = _build_inception_model(device)
            self._inception_device = device
        return self._inception_model

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_best_metric(self, metric_name: str, mode: str = 'min'):
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None, None
        values = self.metrics[metric_name]
        best_idx = np.argmin(values) if mode == 'min' else np.argmax(values)
        best_value = values[best_idx]
        best_epoch = self.metrics['epoch'][best_idx] if len(self.metrics['epoch']) > best_idx else best_idx
        return best_value, best_epoch

    def save(self, filename: str = 'metrics.pt'):
        save_path = self.log_dir / filename
        torch.save(self.metrics, save_path)
        print(f"Saved metrics to {save_path}")

    def load(self, filename: str = 'metrics.pt'):
        load_path = self.log_dir / filename
        if load_path.exists():
            self.metrics = torch.load(load_path)
            print(f"Loaded metrics from {load_path}")
        else:
            print(f"No metrics file found at {load_path}")