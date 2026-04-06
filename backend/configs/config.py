# Pure data configuration for DamageDiffusion training
# All settings are defined here as simple data attributes

from pathlib import Path


class Config:
    """Base configuration for training mask-conditioned DDPM.

    This config contains only data - no methods or logic.
    Hardware detection is handled separately in core/hardware.py

    Using regular class (not dataclass) for simpler inheritance.
    """

    # Paths
    data_root = "../data/crack_segmentation_dataset"
    train_dir = "train"
    test_dir = "test"
    output_dir = "./outputs"
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    sample_dir = "./samples"

    # Dataset
    image_size = 128
    train_val_split = 0.9
    random_seed = 42
    subset_ratio = 1.0  # Use full dataset by default

    # Model architecture
    in_channels = 4
    out_channels = 3
    layers_per_block = 2
    block_out_channels = (128, 256, 512, 512)  # 45M params
    down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

    # Diffusion
    num_train_timesteps = 1000
    beta_schedule = "linear"

    # Training hyperparameters
    train_batch_size = 16
    eval_batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500

    # Optimization
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    use_ema = True
    ema_decay = 0.9999

    # Mixed precision
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # Checkpointing
    save_checkpoint_epochs = 5
    save_top_k = 3
    resume_from_checkpoint = None

    # Logging and evaluation
    log_every_steps = 100
    eval_every_epochs = 5
    generate_samples_every_epochs = 5
    num_inference_steps = 50
    num_samples_to_generate = 8

    # Evaluation metrics
    compute_fid = True
    fid_every_epochs = 10
    num_fid_samples = 1000

    # Data augmentation
    horizontal_flip = True
    random_rotation = False

    # Hardware settings
    num_workers = 4
    pin_memory = True
    device = "cuda"

    def __init__(self):
        """Create output directories on initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)


class FastConfig(Config):
    """Fast training configuration for rapid iteration (2-4 hours).

    Optimizations:
    - 20% data subset
    - 30 epochs
    - Smaller model (12M params vs 45M)
    - Fewer timesteps (100 vs 1000)
    """

    # Override parent values
    subset_ratio = 0.2
    num_epochs = 30
    block_out_channels = (64, 128, 256, 256)  # 12M params
    num_train_timesteps = 100
    save_checkpoint_epochs = 10
