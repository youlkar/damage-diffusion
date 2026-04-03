# training configuration for the DamageDiffusion model

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # configuration for training mask-conditioned DDPM

    # data paths
    data_root: str = "../data/crack_segmentation_dataset"
    train_dir: str = "train"
    test_dir: str = "test"

    # output paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    sample_dir: str = "./samples"

    # dataset settings
    image_size: int = 128  # Resize from 448x448 to 128x128
    train_val_split: float = 0.9  # 90% train, 10% validation
    random_seed: int = 42

    # model architecture
    in_channels: int = 4  # RGB (3) + Binary mask (1)
    out_channels: int = 3  # Predict RGB noise only
    layers_per_block: int = 2
    block_out_channels: tuple = (128, 256, 512, 512)  # Default
    # fast version: (64, 128, 256, 256) - 4x fewer params, 2x faster
    down_block_types: tuple = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    )
    up_block_types: tuple = (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )

    # diffusion settings
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"  # or "scaled_linear", "squaredcos_cap_v2"

    # training hyperparameters
    train_batch_size: int = 16  # Reduce to 8 for M1, keep 16 for P100
    eval_batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500

    # optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_ema: bool = True  # Exponential Moving Average for stable generation
    ema_decay: float = 0.9999

    # mixed precision (P100 supports fp16, M1 uses fp32)
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # checkpointing
    save_checkpoint_epochs: int = 5  # save every 5 epochs
    save_top_k: int = 3  # keep top 3 checkpoints based on validation loss
    resume_from_checkpoint: str = None  # path to checkpoint to resume from

    # logging & evaluation
    log_every_steps: int = 100
    eval_every_epochs: int = 5
    generate_samples_every_epochs: int = 5
    num_inference_steps: int = 50  # for fast sampling during training (full: 1000)
    num_samples_to_generate: int = 8  # number of samples to generate for visualization

    # evaluation metrics
    compute_fid: bool = True
    fid_every_epochs: int = 10
    num_fid_samples: int = 1000  # number of samples for FID calculation

    # data augmentation
    horizontal_flip: bool = True
    random_rotation: bool = False  # Disabled to maintain mask alignment

    # hardware settings
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True

    # device detection (auto-detect CUDA, MPS, or CPU)
    @property
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @property
    def is_distributed(self):
        return torch.cuda.device_count() > 1

    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)


# Preset configurations for different environments
class KaggleP100Config(TrainingConfig):
    # optimized for Kaggle P100 GPU (16GB VRAM)
    train_batch_size: int = 16
    mixed_precision: str = "fp16"
    num_workers: int = 4


class MacBookM1Config(TrainingConfig):
    # optimized for MacBook M1 (limited VRAM, CPU fallback)
    train_batch_size: int = 4  # Smaller batch for limited memory
    mixed_precision: str = "no"  # M1 MPS doesn't fully support fp16 yet
    num_workers: int = 2
    num_epochs: int = 10  # shorter training for local testing


class FastTrainingConfig(TrainingConfig):
    # ultra-fast configuration for rapid iteration (2-4 hour training)
    # subset training
    subset_ratio: float = 0.2  # use only 20% of data
    
    # reduced epochs
    num_epochs: int = 30
    
    # smaller, faster model (4x fewer parameters)
    block_out_channels: tuple = (64, 128, 256, 256)
    
    # fewer diffusion timesteps
    num_train_timesteps: int = 100
    
    # aggressive settings for speed
    train_batch_size: int = 16  # Can be higher on GPU
    save_checkpoint_epochs: int = 10  # Save less frequently
    
    # faster speed settings
    # image_size: int = 64  # Lower resolution
