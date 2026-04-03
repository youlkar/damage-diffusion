"""
Main training script for Mask-Conditioned DDPM (DamageDiffusion).

This script trains a diffusion model to generate realistic crack images
conditioned on binary mask inputs for infrastructure damage assessment.

Usage:
    # Train on Kaggle P100:
    python train.py --config kaggle

    # Train on MacBook M1:
    python train.py --config macbook

    # Resume from checkpoint:
    python train.py --resume ./checkpoints/checkpoint_epoch_50.pt

    # Custom config:
    python train.py --batch_size 32 --num_epochs 200 --learning_rate 2e-4
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from configs.train_config import TrainingConfig, KaggleP100Config, MacBookM1Config, FastTrainingConfig
from configs.colab_config import ColabConfig, ColabT4Config, ColabV100Config, ColabA100Config, ColabFastConfig, ColabFullFastConfig, ColabLatentFastConfig
from data.dataset import get_dataloaders
from models.diffusion import MaskConditionedDDPM
from models.latent_diffusion import LatentDiffusionModel
from utils.training import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Mask-Conditioned DDPM')

    # Configuration presets
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        choices=['default', 'kaggle', 'macbook', 'fast', 'colab', 'colab-t4', 'colab-v100', 'colab-a100', 'colab-fast', 'colab-full-fast', 'colab-latent-fast'],
        help='Configuration preset (default, kaggle, macbook, fast, colab, colab-t4, colab-v100, colab-a100, colab-fast, colab-full-fast, colab-latent-fast)'
    )

    # Data arguments
    parser.add_argument('--data_root', type=str, default='../data/crack_segmentation_dataset',
                       help='Path to dataset root directory')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size for training')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--mixed_precision', type=str, default=None,
                       choices=['fp16', 'bf16', 'no'],
                       help='Mixed precision training')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def setup_config(args):
    """Setup configuration based on arguments."""
    # Select base configuration
    if args.config == 'kaggle':
        config = KaggleP100Config()
        print("Using Kaggle P100 configuration")
    elif args.config == 'macbook':
        config = MacBookM1Config()
        print("Using MacBook M1 configuration")
    elif args.config == 'fast':
        config = FastTrainingConfig()
        print("Using FAST training configuration")
        print("  - 20% data subset")
        print("  - 30 epochs")
        print("  - Smaller model")
        print("  - 100 timesteps")
    elif args.config == 'colab':
        config = ColabConfig()
        print("Using Google Colab configuration with auto-detected GPU")
    elif args.config == 'colab-t4':
        config = ColabT4Config()
        print("Using Google Colab T4 configuration")
    elif args.config == 'colab-v100':
        config = ColabV100Config()
        print("Using Google Colab V100 configuration")
    elif args.config == 'colab-a100':
        config = ColabA100Config()
        print("Using Google Colab A100 configuration")
    elif args.config == 'colab-fast':
        config = ColabFastConfig()
        print("Using Google Colab FAST configuration")
        print("  - 30% data subset")
        print("  - 40 epochs")
        print("  - Smaller model")
        print("  - 100 timesteps")
    elif args.config == 'colab-full-fast':
        config = ColabFullFastConfig()
        print("Using Google Colab FULL+FAST configuration")
        print("  - 100% data")
        print("  - 50 epochs")
        print("  - Smaller model")
        print("  - 200 timesteps")
    elif args.config == 'colab-latent-fast':
        config = ColabLatentFastConfig()
        print("Using Google Colab LATENT DIFFUSION configuration")
        print("  - Latent Diffusion Model based on Stable Diffusion")
        print("  - 100% data")
        print("  - 40 epochs")
        print("  - Pre-trained VAE frozen")
        print("  - Diffusion in 16x16x4 latent space")
        print("  - 100 timesteps")
    else:
        config = TrainingConfig()
        print("Using default configuration")

    # Override with command line arguments
    if args.data_root:
        config.data_root = args.data_root
    if args.image_size:
        config.image_size = args.image_size
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.mixed_precision:
        config.mixed_precision = args.mixed_precision
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.num_workers:
        config.num_workers = args.num_workers

    config.random_seed = args.seed

    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup configuration
    config = setup_config(args)

    # Set random seed
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        # Optimize CUDNN for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Print configuration
    print("\n" + "="*60)
    print("DamageDiffusion Training Configuration")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Data root: {config.data_root}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Log dir: {config.log_dir}")
    print("="*60 + "\n")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nInitializing model...")
    if hasattr(config, 'use_latent_diffusion') and config.use_latent_diffusion:
        # Use Latent Diffusion Model
        print("Using Latent Diffusion Model (LDM)")
        model = LatentDiffusionModel(
            vae_model_name=config.vae_model,
            latent_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            num_train_timesteps=config.num_train_timesteps,
            device=config.device,
        )
    else:
        # Use standard pixel-space DDPM
        print("Using pixel-space DDPM")
        model = MaskConditionedDDPM(config)
    
    # Compile model for PyTorch 2.0+ speedup
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            if hasattr(config, 'use_latent_diffusion') and config.use_latent_diffusion:
                # Only compile U-Net for LDM (VAE is frozen)
                model.unet = torch.compile(model.unet, mode='reduce-overhead')
            else:
                model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        except Exception as e:
            print(f"torch.compile failed (continuing without): {e}")
    else:
        print("torch.compile not available (PyTorch < 2.0)")

    # Create optimizer
    if hasattr(config, 'use_latent_diffusion') and config.use_latent_diffusion:
        # Only train U-Net parameters (VAE is frozen)
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=config,
        device=config.device,
    )

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint('interrupted_checkpoint.pt')
        print("Saved checkpoint before exiting")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint('error_checkpoint.pt')
        print("Saved checkpoint before exiting")
        raise

    print("\nTraining completed successfully!")

    # Test final model
    print("\nEvaluating on test set...")
    test_loss = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")

    # Generate final samples
    print("\nGenerating final samples...")
    images, masks, generated = trainer.generate_samples(num_samples=16)

    from utils.visualization import visualize_samples
    visualize_samples(
        images, masks, generated,
        num_samples=8,
        save_path=f"{config.sample_dir}/final_samples.png"
    )

    print(f"\nAll outputs saved to:")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"  Logs: {config.log_dir}")
    print(f"  Samples: {config.sample_dir}")

    print("\nTo view training progress with TensorBoard:")
    print(f"  tensorboard --logdir={config.log_dir}")


if __name__ == '__main__':
    main()
