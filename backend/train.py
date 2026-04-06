# main training script for DDPM
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from configs.train_config import TrainingConfig, FastTrainingConfig
from data.dataset import get_dataloaders
from models.diffusion import MaskConditionedDDPM
from utils.training import Trainer
import traceback
from utils.visualization import visualize_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask-Conditioned DDPM')

    # config preset
    parser.add_argument('--config', type=str, default='default', choices=['default', 'fast'],
                       help='Configuration preset: default (full quality), fast (2-4h training)')

    # data args
    parser.add_argument('--data_root', type=str, help='Path to dataset root directory')
    parser.add_argument('--image_size', type=int, help='Image size for training')

    # training args
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'bf16', 'no'],
                       help='Mixed precision training')

    # checkpoint args
    parser.add_argument('--resume', type=str, help='Path to checkpoint to continue from')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, help='Directory for TensorBoard logs')

    # hardware args
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def apply_cli_overrides(config, args):
    overrides = {
        'data_root': 'data_root',
        'image_size': 'image_size',
        'batch_size': 'train_batch_size',
        'num_epochs': 'num_epochs',
        'learning_rate': 'learning_rate',
        'mixed_precision': 'mixed_precision',
        'resume': 'resume_from_checkpoint',
        'checkpoint_dir': 'checkpoint_dir',
        'log_dir': 'log_dir',
        'num_workers': 'num_workers',
    }

    for arg_name, config_attr in overrides.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config, config_attr, arg_value)

    config.random_seed = args.seed
    return config


def detect_hardware(config):
    """Auto-detect hardware and configure settings."""
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)

        # VRAM-based batch sizing
        config.train_batch_size = max(4, min(48, int(vram_gb * 1.5)))
        config.eval_batch_size = config.train_batch_size * 2
        config.device = "cuda"

        print(f"CUDA GPU Detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        print(f"Config applied: batch_size={config.train_batch_size}, eval_batch_size={config.eval_batch_size}")

    elif torch.backends.mps.is_available():
        config.train_batch_size = 4
        config.eval_batch_size = 8
        config.device = "mps"
        config.mixed_precision = "no"
        config.num_workers = 2

        print("Apple Silicon Detected (MPS)")
        print(f"Config applied: batch_size={config.train_batch_size}, mixed_precision=no")

    else:
        config.train_batch_size = 2
        config.eval_batch_size = 4
        config.device = "cpu"
        config.mixed_precision = "no"
        config.num_workers = 2

        print("CPU Detected")
        print(f"Config applied: batch_size={config.train_batch_size}, mixed_precision=no")

    return config


def create_model(config):
    """Create and optimize model."""
    print("Initializing model...")
    model = MaskConditionedDDPM(config)

    # channels-last for GPU
    if config.device == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("Model converted to channels-last format")

    # torch.compile optimization
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        except Exception as e:
            print(f"torch.compile failed (continuing without): {e}")

    # print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    return model


def main():
    args = parse_args()

    # create config
    if args.config == 'fast':
        config = FastTrainingConfig()
        print("\n" + "-"*50)
        print("Fast training config approx 2-4 hrs")
        print("-"*50)
        print("20% data subset")
        print("30 epochs")
        print("Smaller model (12M params)")
        print("100 timesteps")
        print("-"*50 + "\n")
    else:
        config = TrainingConfig()
        print("\n" + "-"*50)
        print("Full regular training config")
        print("-"*50)
        print("Auto detect hardware")
        print("Batch sizing based on vram")
        print("-"*50 + "\n")

    # apply CLI overrides
    config = apply_cli_overrides(config, args)

    # auto-detect hardware
    config = detect_hardware(config)

    # random seeds
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # config summary
    print("\nDamageDiffusion Training Configuration:")
    print("-"*50)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Data root: {config.data_root}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Log dir: {config.log_dir}")

    # load datasets
    print("Loading datasets")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # create model and optimizer
    model = create_model(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=config,
        device=config.device,
    )

    # start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint('interrupted_checkpoint.pt')
        print("Saved checkpoint before exiting")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
        trainer.save_checkpoint('error_checkpoint.pt')
        print("Saved checkpoint before exiting")
        raise

    print("\nTraining completed successfully!")

    # evaluate on test set
    print("\nEvaluating on test set")
    test_loss = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")

    # generate final samples
    print("\nGenerating final samples")
    images, masks, generated = trainer.generate_samples(num_samples=16)

    visualize_samples(images, masks, generated, num_samples=8, save_path=f"{config.sample_dir}/final_samples.png")

    print(f"\nFile save directories:")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Logs: {config.log_dir}")
    print(f"Samples: {config.sample_dir}")
    print(f"\nTo view training progress with TensorBoard:")
    print(f"tensorboard --logdir={config.log_dir}")


if __name__ == '__main__':
    main()
