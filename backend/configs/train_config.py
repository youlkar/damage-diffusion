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
    num_inference_steps = 50
    num_samples_to_generate = 8

    # evaluation metrics
    compute_fid = True
    fid_every_epochs = 10
    num_fid_samples = 1000

    # data augmentation
    horizontal_flip = True
    random_rotation = False

    # hardware settings
    num_workers = 4
    pin_memory = True
    device = "cuda"
    use_compile = True  # enable torch.compile by default

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

        # disable torch.compile to save memory
        self.use_compile = False


class FastTrainingConfig(TrainingConfig):
    # fast training configuration (2-4 hours)

    def __init__(self):
        super().__init__()

        # override for fast training
        self.subset_ratio = 0.2
        self.num_epochs = 30
        self.block_out_channels = (64, 128, 256, 256)
        self.num_train_timesteps = 100
        self.save_checkpoint_epochs = 10

        # reduce memory usage during sample generation
        self.num_samples_to_generate = 4
        self.num_inference_steps = 25

        # disable torch.compile to save memory (CUDA graphs use too much)
        self.use_compile = False
