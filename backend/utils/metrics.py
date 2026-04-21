# Evaluation metrics for generative model quality assessment.
# Includes FID (Fréchet Inception Distance) and IoU calculations.

import torch
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def compute_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = 'cuda',
) -> float:
    # initialize inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    # normalize images from [-1, 1] to [0, 1] then to inception's expected range
    def preprocess_for_inception(images):
        # denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0

        # resize to 299x299 (inception input size)
        resize = transforms.Resize((299, 299))
        images = torch.stack([resize(img) for img in images])

        # normalize to inception's expected range
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        images = torch.stack([normalize(img) for img in images])
        return images

    # extract features
    with torch.no_grad():
        real_images_proc = preprocess_for_inception(real_images).to(device)
        gen_images_proc = preprocess_for_inception(generated_images).to(device)

        real_features = []
        gen_features = []

        # process in batches to avoid OOM
        batch_size = 32
        for i in range(0, len(real_images_proc), batch_size):
            batch = real_images_proc[i:i+batch_size]
            feat = inception_model(batch)[0].squeeze(-1).squeeze(-1)
            real_features.append(feat.cpu().numpy())

        for i in range(0, len(gen_images_proc), batch_size):
            batch = gen_images_proc[i:i+batch_size]
            feat = inception_model(batch)[0].squeeze(-1).squeeze(-1)
            gen_features.append(feat.cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)

    # calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid

def compute_fid_kid_scores(real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = 'cuda'
) -> Tuple[float, float]:
    
    # normalize images from [-1, 1] to [0, 1] then to inception's expected range
    def preprocess_for_inception(images):
        # denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0

        # resize to 299x299 (inception input size)
        resize = transforms.Resize((299, 299))
        images = torch.stack([resize(img) for img in images])

        # normalize to inception's expected range
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        images = torch.stack([normalize(img) for img in images])
        return images
    
    with torch.no_grad():
        real_images_proc = preprocess_for_inception(real_images).to(device)
        gen_images_proc = preprocess_for_inception(generated_images).to(device)
    
        # compute FID
        print("Computing FID...")
        fid = FrechetInceptionDistance(feature=64)
        fid.to(device)
        fid.update(real_images_proc, real=True)
        fid.update(gen_images_proc, real=False)
        fid_tens = fid.compute()
        fid_score = fid_tens[0]
        fid.reset()
        
        # compute KID
        print("Computing KID...")
        kid = KernelInceptionDistance()
        kid.to(device)
        kid.update(real_images_proc, real=True)
        kid.update(gen_images_proc, real=False)
        kid_tens = kid.compute()
        kid_mean = kid_tens[0]
        kid.reset()
        return fid_score, kid_mean

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
    # binarize masks
    pred_binary = (pred_mask > threshold).float()
    true_binary = (true_mask > threshold).float()

    # calculate intersection and union
    intersection = (pred_binary * true_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + true_binary.sum(dim=(-2, -1)) - intersection

    # avoid division by zero by adding a small epsilon
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

    accuracy = (correct / total).item()
    return accuracy


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
    # track and log metrics during training

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'fid_score': [],
            'kid_score': [],
            'learning_rate': [],
            'epoch': [],
        }

    def update(self, **kwargs):
        # update metrics
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_best_metric(self, metric_name: str, mode: str = 'min'):
        # get best metric value and epoch
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None, None

        values = self.metrics[metric_name]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

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
