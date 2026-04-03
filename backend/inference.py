# inference script for generating crack images from masks.


import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import sys
from models.diffusion import MaskConditionedDDPM
from models.latent_diffusion import LatentDiffusionModel
from configs.train_config import TrainingConfig
from configs.colab_config import ColabLatentFastConfig
from utils.visualization import save_samples, denormalize

sys.path.append(str(Path(__file__).parent))


def load_model(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    config_dict = checkpoint.get('config', {})

    # Check if it's a latent diffusion model
    use_latent_diffusion = config_dict.get('use_latent_diffusion', False)

    if use_latent_diffusion:
        # Load as Latent Diffusion Model
        config = ColabLatentFastConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        model = LatentDiffusionModel(
            vae_model_name=config.vae_model,
            latent_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            num_train_timesteps=config.num_train_timesteps,
            device=device,
        )
        # Load only U-Net weights (VAE is pre-trained and frozen)
        model.unet.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded Latent Diffusion Model")
    else:
        # Load as standard pixel-space DDPM
        config = TrainingConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        model = MaskConditionedDDPM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pixel-space DDPM")

    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")

    return model, config


# load and preprocess the mask
def load_mask(mask_path: str, image_size: int = 128):
    mask = Image.open(mask_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    mask = transform(mask)
    mask = (mask > 0.5).float()

    return mask.unsqueeze(0)


@torch.no_grad()
def generate_from_mask(
    model,
    mask: torch.Tensor,
    num_samples: int = 1,
    num_inference_steps: int = 50,
    device: str = 'cuda',
):
    # Repeat mask for multiple samples
    masks = mask.repeat(num_samples, 1, 1, 1).to(device)

    # Check if using latent diffusion
    is_latent_diffusion = isinstance(model, LatentDiffusionModel)

    # Generate
    print(f"Generating {num_samples} samples with {num_inference_steps} steps...")

    if is_latent_diffusion:
        # LATENT DIFFUSION: Encode mask and sample
        # Need to normalize mask to [-1, 1] for VAE encoding
        mask_normalized = masks * 2.0 - 1.0
        mask_latents = model.encode_masks(masks)
        generated = model.sample(mask_latents, num_inference_steps=num_inference_steps)
    else:
        # PIXEL-SPACE DDPM
        generated = model.generate(masks, num_inference_steps=num_inference_steps)

    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate crack images from masks')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mask_path', type=str, default=None,
                       help='Path to single mask image')
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='Path to directory of masks (batch processing)')
    parser.add_argument('--output_dir', type=str, default='./generated',
                       help='Output directory for generated images')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate per mask')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of inference steps (50 for fast, 1000 for quality)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device=device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single mask generation
    if args.mask_path:
        print(f"\nGenerating from mask: {args.mask_path}")

        mask = load_mask(args.mask_path, image_size=config.image_size)
        generated = generate_from_mask(
            model,
            mask,
            num_samples=args.num_samples,
            num_inference_steps=args.num_steps,
            device=device,
        )

        # Save
        mask_name = Path(args.mask_path).stem
        save_path = output_dir / f"{mask_name}_generated.png"
        save_samples(generated, str(save_path), nrow=min(args.num_samples, 8))

        print(f"Saved generated images to {save_path}")

    # Batch generation from directory
    elif args.mask_dir:
        mask_dir = Path(args.mask_dir)
        mask_files = sorted(mask_dir.glob('*.png')) + sorted(mask_dir.glob('*.jpg'))

        print(f"\nFound {len(mask_files)} masks in {mask_dir}")

        for i, mask_file in enumerate(mask_files):
            print(f"\nProcessing [{i+1}/{len(mask_files)}]: {mask_file.name}")

            mask = load_mask(str(mask_file), image_size=config.image_size)
            generated = generate_from_mask(
                model,
                mask,
                num_samples=args.num_samples,
                num_inference_steps=args.num_steps,
                device=device,
            )

            # Save
            save_path = output_dir / f"{mask_file.stem}_generated.png"
            save_samples(generated, str(save_path), nrow=min(args.num_samples, 8))

            print(f"Saved to {save_path}")

    else:
        print("Error: Must provide either --mask_path or --mask_dir")
        return

    print(f"\nPASS: Generation complete! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
