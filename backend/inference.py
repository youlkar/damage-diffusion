# inference script for generating crack images from masks.


import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import sys
from models.diffusion import MaskConditionedDDPM
from configs.train_config import TrainingConfig
from utils.visualization import save_samples, denormalize

sys.path.append(str(Path(__file__).parent))


def load_model(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate the EXACT config used during training
    config = TrainingConfig()
    if 'config' in checkpoint:
        # Apply all saved config values
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"Loaded training config:")
        print(f"Model channels: {config.block_out_channels}")
        print(f"Timesteps: {config.num_train_timesteps}")
        print(f"Image size: {config.image_size}")
    else:
        print("WARNING: No config found in checkpoint, using defaults")

    # Create model
    model = MaskConditionedDDPM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    guidance_scale: float = 1.0,
    device: str = 'cuda',
):
    masks = mask.repeat(num_samples, 1, 1, 1).to(device)

    if guidance_scale > 1.0:
        print(f"Generating {num_samples} samples with {num_inference_steps} steps "
              f"and CFG guidance_scale={guidance_scale}...")
        generated = model.generate_cfg(
            masks,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    else:
        print(f"Generating {num_samples} samples with {num_inference_steps} steps...")
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
                       help='Number of inference steps (50 for fast, 500 for quality)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='CFG guidance scale (1.0=off, 3-7=recommended for crack visibility)')
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
            guidance_scale=args.guidance_scale,
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
                guidance_scale=args.guidance_scale,
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
