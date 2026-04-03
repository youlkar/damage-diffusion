

from .train_config import TrainingConfig


class ColabConfig(TrainingConfig):
    def __init__(self, project_dir: str = "/content/drive/MyDrive/DamageDiffusion"):
        super().__init__()

        # update paths for Colab/Drive
        self.data_root = f"{project_dir}/data/crack_segmentation_dataset"
        self.output_dir = f"{project_dir}/outputs"
        self.checkpoint_dir = f"{project_dir}/checkpoints"
        self.log_dir = f"{project_dir}/logs"
        self.sample_dir = f"{project_dir}/samples"

        # colab-optimized settings
        self.num_workers = 2  # Colab works better with fewer workers
        self.pin_memory = True

        # auto-adjust batch size based on GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

            if 'T4' in gpu_name:
                self.train_batch_size = 12
                self.eval_batch_size = 24
                print(f"Detected {gpu_name} - using batch_size=12")
            elif 'V100' in gpu_name:
                self.train_batch_size = 24
                self.eval_batch_size = 48
                print(f"Detected {gpu_name} - using batch_size=24")
            elif 'A100' in gpu_name or 'A10' in gpu_name:
                self.train_batch_size = 48
                self.eval_batch_size = 64
                print(f"Detected {gpu_name} - using batch_size=48")
            else:
                self.train_batch_size = 16
                self.eval_batch_size = 32
                print(f"Detected {gpu_name} - using batch_size=16")
        else:
            print("No GPU detected!")
            self.train_batch_size = 8


class ColabT4Config(ColabConfig):
    # optimized for Colab T4 GPU (16GB VRAM)
    train_batch_size: int = 12
    eval_batch_size: int = 24


class ColabV100Config(ColabConfig):
    # optimized for Colab V100 GPU (16GB VRAM)
    train_batch_size: int = 24
    eval_batch_size: int = 48


class ColabA100Config(ColabConfig):
    # optimized for Colab A100 GPU (40GB VRAM)
    train_batch_size: int = 48
    eval_batch_size: int = 64

    # can use larger model for A100
    image_size: int = 128  # Could go to 256 if needed


class ColabFastConfig(ColabConfig):
    """
    Ultra-fast Colab training (2-3 hours total).
    Optimized for rapid iteration while maintaining quality.
    """
    def __init__(self, project_dir: str = "/content/drive/MyDrive/DamageDiffusion"):
        super().__init__(project_dir)

        # Speed optimizations
        self.subset_ratio = 0.3  # Use 30% of data (~2,880 samples)
        self.num_epochs = 40  # Reduced from 100

        # Smaller, faster model (4x fewer parameters)
        self.block_out_channels = (64, 128, 256, 256)

        # Fewer diffusion timesteps (10x faster per step)
        self.num_train_timesteps = 100  # vs 1000

        # Aggressive batch size for speed
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'T4' in gpu_name:
                self.train_batch_size = 16
            elif 'V100' in gpu_name:
                self.train_batch_size = 32
            elif 'A100' in gpu_name:
                self.train_batch_size = 64
            else:
                self.train_batch_size = 24

        # More frequent validation/sampling for monitoring
        self.validate_every_epochs = 5
        self.generate_samples_every_epochs = 5
        self.save_checkpoint_epochs = 10


class ColabFullFastConfig(ColabConfig):
    """
    Full dataset + optimized training (8-12 hours total).
    Uses 100% of data with speed optimizations.
    Best balance of quality and training time.
    """
    def __init__(self, project_dir: str = "/content/drive/MyDrive/DamageDiffusion"):
        super().__init__(project_dir)

        # Use ALL data (no subset)
        self.subset_ratio = 1.0  # 100% of data (~9,603 samples)

        # Reduced epochs (sufficient for convergence)
        self.num_epochs = 50  # vs 100 (still gets good results)

        # Smaller, faster model (4x fewer parameters, minimal quality loss)
        self.block_out_channels = (64, 128, 256, 256)

        # Reduced timesteps (5x faster, still good quality)
        self.num_train_timesteps = 200  # vs 1000

        # Optimized batch size for speed
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'T4' in gpu_name:
                self.train_batch_size = 20
                self.mixed_precision = "fp16"  # Mixed precision for speed
            elif 'V100' in gpu_name:
                self.train_batch_size = 32
                self.mixed_precision = "fp16"
            elif 'A100' in gpu_name:
                self.train_batch_size = 64
                self.mixed_precision = "fp16"
            else:
                self.train_batch_size = 24
                self.mixed_precision = "fp16"

        # Checkpoint settings
        self.validate_every_epochs = 5
        self.generate_samples_every_epochs = 5
        self.save_checkpoint_epochs = 10


class ColabLatentFastConfig(ColabConfig):
    """
    Latent Diffusion Model (LDM) configuration for accelerated training.

    Uses pre-trained VAE for faster training over pixel-space diffusion.
    Trains DDPM in compressed 16x16x4 latent space instead of 128x128x3 pixels.

    Architecture: Based on Stable Diffusion (Rombach et al., 2022)
    - Pre-trained VAE: stabilityai/sd-vae-ft-mse (frozen)
    - U-Net: Denoising in latent space
    - Output: Decoded to 128x128 RGB images
    """
    def __init__(self, project_dir: str = "/content/drive/MyDrive/DamageDiffusion"):
        super().__init__(project_dir)

        # === LATENT DIFFUSION SETTINGS ===
        self.use_latent_diffusion = True
        self.vae_model = "stabilityai/sd-vae-ft-mse"  # Pre-trained VAE
        self.vae_scaling_factor = 0.18215  # Stable Diffusion scaling

        # Latent space dimensions (128 / 8 = 16)
        self.latent_channels = 4
        self.latent_size = 16

        # Original image size (for data loading)
        self.image_size = 128

        # === TRAINING OPTIMIZATIONS ===
        # Full dataset
        self.subset_ratio = 1.0  # 100% of data (~9,603 samples)

        # Fewer epochs needed (latent space converges faster)
        self.num_epochs = 40

        # Can use more timesteps since latent is faster
        self.num_train_timesteps = 100

        # Smaller U-Net for latent space (4x fewer params than pixel-space)
        self.block_out_channels = (64, 128, 256, 256)

        # === GPU-SPECIFIC SETTINGS ===
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

            if 'T4' in gpu_name:
                self.train_batch_size = 24  # Can use larger batch in latent space
                self.eval_batch_size = 32
                self.mixed_precision = "fp16"
                print(f"Detected {gpu_name} - Latent mode batch_size=24")

            elif 'V100' in gpu_name:
                self.train_batch_size = 40
                self.eval_batch_size = 64
                self.mixed_precision = "fp16"
                print(f"Detected {gpu_name} - Latent mode batch_size=40")

            elif 'A100' in gpu_name:
                self.train_batch_size = 64
                self.eval_batch_size = 96
                self.mixed_precision = "fp16"
                print(f"Detected {gpu_name} - Latent mode batch_size=64")

            else:
                self.train_batch_size = 32
                self.eval_batch_size = 48
                self.mixed_precision = "fp16"
                print(f"Detected {gpu_name} - Latent mode batch_size=32")
        else:
            print("No GPU detected - Latent diffusion requires GPU!")
            self.train_batch_size = 8

        # === CHECKPOINT & MONITORING ===
        self.validate_every_epochs = 5
        self.generate_samples_every_epochs = 5
        self.save_checkpoint_epochs = 10

        # Inference settings
        self.num_inference_steps = 50  # Faster inference in latent space
