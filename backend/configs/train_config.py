# training configuration for the DamageDiffusion model

import torch
from dataclasses import dataclass
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
    image_size = 128  # resize from 448x448 to 128x128
    train_val_split = 0.9  # 90% train, 10% val
    random_seed = 42

    # model architecture
    in_channels = 4  # 3 rgb + 1 mask
    out_channels = 3  # predict rgb noise only
    layers_per_block = 2
    block_out_channels = (128, 256, 512, 512)  # default 45 million params

    # fast version: (64, 128, 256, 256), 12mil params
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

    # mixed precision - automatically configured for cpu/mps
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0


    # logging & evaluation
    log_every_steps = 100
    eval_every_epochs = 5
    generate_samples_every_epochs = 5
    num_inference_steps = 50
    num_samples_to_generate = 8

    # evaluation metrics
    compute_fid = True
    fid_every_epochs = 10
    num_fid_samples = 1000

    # data augmentation
    horizontal_flip = True
    random_rotation = False  # disabled to maintain mask alignment

    # hardware settings
    num_workers = 4  # dataloader workers
    pin_memory = True
    auto_detect_hardware = True  # enable automatic hardware detection

    def is_distributed(self):
        return torch.cuda.device_count() > 1

    # device detection - detect cuda, mps, or cpu
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # batch size based on vram
    def auto_detect_batch_size(self):
        if torch.cuda.is_available():
            
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)                                                                                                                                                                  
    
            # adjust batch size based on vram 
            self.train_batch_size = max(4, min(48, int(vram_gb * 1.5)))
            self.eval_batch_size = self.train_batch_size * 2
                                                                                                                                                                                                                        
            print(f"CUDA GPU Detected: {gpu_name} ({vram_gb:.1f}gb VRAM)")
            print(f"Config applied: batch_size={self.train_batch_size}, eval_batch_size={self.eval_batch_size}")  

        # settings for apple silicon
        elif torch.backends.mps.is_available():
            self.train_batch_size = 4
            self.eval_batch_size = 8
            self.mixed_precision = "no"
            self.num_workers = 2

            print("Apple silicon detected")
            print(f"Config applied: batch_size={self.train_batch_size}, mixed_precision=no")

        # settings for cpu - fallback setting
        else:
            self.train_batch_size = 2
            self.eval_batch_size = 4
            self.mixed_precision = "no"
            self.num_workers = 2

            print("CPU detected")
            print(f"Config applied: batch_size={self.train_batch_size}, mixed_precision=no")

    def __post_init__(self):
        # auto-detect hardware and set batch sizes
        if self.auto_detect_hardware:
            self._auto_detect_batch_size()

        # output directories for checkpoints, logs, and samples
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)


# fast training config
class FastTrainingConfig(TrainingConfig):
    # faster training configuration for quick iteration - approx 2-4 hours
    # batch sizing auto detected by parent class

    # data subset for speed, smaller params, less frequent checkpoints, fewer timesteps
    subset_ratio = 0.2
    num_epochs = 30
    block_out_channels = (64, 128, 256, 256)
    num_train_timesteps = 100
    save_checkpoint_epochs = 10
