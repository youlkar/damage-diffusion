

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
