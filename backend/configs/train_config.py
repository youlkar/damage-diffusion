# training configuration for the DamageDiffusion model

import torch
from pathlib import Path


class TrainingConfig:
    # universal training configuration for different hardware

    # data and output paths
    data_root = "../data/crack_segmentation_dataset"
    train_dir = "train"
    test_dir = "test"
    output_dir = "./outputs"
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    sample_dir = "./samples"

    # checkpoint saving
    save_checkpoint_epochs = 5
    save_top_k = 3
    resume_from_checkpoint = None

    # dataset settings
    image_size = 128
    train_val_split = 0.9
    random_seed = 42

    # model architecture
    in_channels = 4
    out_channels = 3
    layers_per_block = 2
    block_out_channels = (128, 256, 512, 512)
    down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

    # diffusion settings
    num_train_timesteps = 1000
    beta_schedule = "linear"

    # training hyperparameters
    train_batch_size = 16
    eval_batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500

    # optimization
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    use_ema = True
    ema_decay = 0.9999

    # mixed precision
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # logging and evaluation
    log_every_steps = 100
    eval_every_epochs = 5
    generate_samples_every_epochs = 5
    num_inference_steps = 100  # 100 steps: better image quality for reliable FID/KID measurement
    num_samples_to_generate = 8

    # evaluation metrics (KID/FID)
    compute_metrics = True  # Enable KID/FID computation
    metrics_every_epochs = 10  # Compute every 10 epochs
    num_metrics_samples = 2048  # 2048 minimum for statistically reliable FID/KID

    # Enhanced stochastic data augmentation for better KID/FID scores
    horizontal_flip = True
    random_rotation = True
    rotation_degrees = 15  # Random rotation ±15 degrees
    color_jitter = True
    color_jitter_brightness = 0.1
    color_jitter_contrast = 0.1
    random_crop_scale = (0.9, 1.0)  # Random crop 90-100% of image
    noise_injection = True
    noise_injection_std = 0.02  # Add small amount of noise during training

    # hardware settings
    num_workers = 4
    pin_memory = True
    device = "cuda"
    use_compile = False  # Disabled: incompatible with diffusers DDPM scheduler (causes cudagraph partition)

    def __init__(self):
        # create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)


class MediumTrainingConfig(TrainingConfig):
    # medium training configuration (4-6 hours) - validation test
    # Use this to verify crack generation works before committing to full 18hr training

    def __init__(self):
        super().__init__()

        # balanced settings for validation
        self.subset_ratio = 0.5
        self.num_epochs = 50
        self.block_out_channels = (96, 192, 384, 384)
        self.num_train_timesteps = 500
        self.save_checkpoint_epochs = 5

        # moderate sample generation
        self.num_samples_to_generate = 6
        self.num_inference_steps = 50

        # KID/FID metrics for medium config  
        self.compute_metrics = True  # Enable KID/FID computation
        self.metrics_every_epochs = 25  # Compute every 25 epochs
        self.num_metrics_samples = 500  # Use 500 samples (balanced speed vs accuracy)

        # disable torch.compile to save memory
        self.use_compile = False


class FastTrainingConfig(TrainingConfig):
    # fast training configuration (2-4 hours)

    def __init__(self):
        super().__init__()

        # IMPROVED fast training - still fast but actually learns cracks
        self.subset_ratio = 0.5  # Use 50% of data instead of 20%
        self.num_epochs = 50     # Increase to 50 epochs instead of 30
        self.block_out_channels = (96, 192, 384, 384)  # Larger model
        self.num_train_timesteps = 500  # CRITICAL: Use 500 timesteps instead of 100
        self.save_checkpoint_epochs = 10

        # Enhanced stochastic training for better KID/FID scores
        self.random_rotation = True  # Enable rotation augmentation
        self.rotation_degrees = 10   # Smaller rotation for crack preservation
        self.color_jitter = True     # Enable color variations
        self.color_jitter_brightness = 0.05  # Subtle brightness changes
        self.color_jitter_contrast = 0.05    # Subtle contrast changes
        self.random_crop_scale = (0.95, 1.0) # Minimal random cropping
        self.noise_injection = True          # Add training noise
        self.noise_injection_std = 0.01      # Very small noise amount

        # Enable KID/FID metrics for stochastic training validation
        self.compute_metrics = True  # Enable KID/FID computation
        self.metrics_every_epochs = 25  # Compute every 25 epochs (2x per training)
        self.num_metrics_samples = 200  # Use 200 samples for faster computation

        # reduce memory usage during sample generation
        self.num_samples_to_generate = 4
        self.num_inference_steps = 50  # Increase inference steps

        # disable torch.compile to save memory (CUDA graphs use too much)
        self.use_compile = False
