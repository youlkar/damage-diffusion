# dataset class for crack segmentation
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class CrackSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 128,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        horizontal_flip: bool = True,
        config = None,  # Add config for enhanced augmentations
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip and split == 'train'
        self.config = config  # Store config for augmentations


        if split in ['train', 'val']:
            self.image_dir = self.root_dir / 'train' / 'images'
            self.mask_dir = self.root_dir / 'train' / 'masks'
        else:  # test
            self.image_dir = self.root_dir / 'test' / 'images'
            self.mask_dir = self.root_dir / 'test' / 'masks'

        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])

        self.valid_pairs = self.validate_pairs()

        print(f"Loaded {len(self.valid_pairs)} {split} samples from {self.image_dir}")

        # resize and normalize images and masks
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.mask_transform = mask_transform

    def validate_pairs(self):
        # validate each image has a corresponding mask
        valid_pairs = []
        for img_file in self.image_files:
            mask_file = img_file  # Masks have the same filename as images
            if os.path.exists(self.mask_dir / mask_file):
                valid_pairs.append(img_file)
            else:
                print(f"Warning: No mask found for {img_file}")
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name = self.valid_pairs[idx]

        # Load image and mask
        image_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        # Enhanced stochastic augmentations for better diversity
        if self.horizontal_flip and torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            
        # Additional augmentations if enabled in config
        if hasattr(self, 'config') and self.config:
            # Random rotation (both image and mask)
            if getattr(self.config, 'random_rotation', False) and torch.rand(1) > 0.5:
                angle = torch.empty(1).uniform_(
                    -self.config.rotation_degrees,
                    self.config.rotation_degrees
                ).item()
                image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = transforms.functional.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
            
            # Color jittering (image only)
            if getattr(self.config, 'color_jitter', False) and torch.rand(1) > 0.5:
                brightness = 1.0 + torch.empty(1).uniform_(
                    -self.config.color_jitter_brightness,
                    self.config.color_jitter_brightness
                ).item()
                contrast = 1.0 + torch.empty(1).uniform_(
                    -self.config.color_jitter_contrast,
                    self.config.color_jitter_contrast
                ).item()
                image = transforms.functional.adjust_brightness(image, brightness)
                image = transforms.functional.adjust_contrast(image, contrast)
            
            # Random crop and resize (both image and mask)
            if getattr(self.config, 'random_crop_scale', None) and torch.rand(1) > 0.5:
                scale_min, scale_max = self.config.random_crop_scale
                scale = torch.empty(1).uniform_(scale_min, scale_max).item()
                
                # Calculate crop size
                h, w = image.shape[-2:]
                crop_h, crop_w = int(h * scale), int(w * scale)
                
                # Random crop position
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                
                # Apply crop and resize back
                image = transforms.functional.crop(image, top, left, crop_h, crop_w)
                mask = transforms.functional.crop(mask, top, left, crop_h, crop_w)
                image = transforms.functional.resize(image, (h, w), interpolation=transforms.InterpolationMode.BILINEAR)
                mask = transforms.functional.resize(mask, (h, w), interpolation=transforms.InterpolationMode.NEAREST)
            
            # Noise injection (image only)
            if getattr(self.config, 'noise_injection', False) and torch.rand(1) > 0.5:
                noise = torch.randn_like(image) * self.config.noise_injection_std
                image = torch.clamp(image + noise, -1, 1)

        return image, mask

    def create_train_val_split(root_dir, train_ratio = 0.9, image_size = 128, random_seed = 42, horizontal_flip = True, config = None):
  
        full_dataset = CrackSegmentationDataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            horizontal_flip=horizontal_flip,
            config=config,  # Pass config for enhanced augmentations
        )

        # calculate split sizes
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        # split dataset
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator
        )

        # create validation dataset with proper transforms (no augmentation)
        val_dataset_no_aug = CrackSegmentationDataset(
            root_dir=root_dir,
            split='val',
            image_size=image_size,
            horizontal_flip=False,
            config=None,  # No augmentations for validation
        )


        val_dataset.dataset = val_dataset_no_aug
        print(f"Split dataset: {train_size} train, {val_size} validation samples")

        return train_dataset, val_dataset

    def get_sample_batch(self, num_samples = 4):
        # get a batch of samples for visualization
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        images, masks = [], []
        for idx in indices:
            img, mask = self[idx]
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)


def get_dataloaders(config):

    # datasets with enhanced stochastic augmentations
    train_dataset, val_dataset = CrackSegmentationDataset.create_train_val_split(
        root_dir=config.data_root,
        train_ratio=config.train_val_split,
        image_size=config.image_size,
        random_seed=config.random_seed,
        horizontal_flip=config.horizontal_flip,
        config=config,  # Pass config for enhanced augmentations
    )

    # apply subset
    if hasattr(config, 'subset_ratio') and config.subset_ratio < 1.0:

        original_size = len(train_dataset)
        subset_size = int(original_size * config.subset_ratio)

        # Use random subset for better diversity
        indices = torch.randperm(original_size)[:subset_size].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

        print(f"Using {config.subset_ratio*100:.0f}% data subset: "
              f"{subset_size:,}/{original_size:,} samples")

    test_dataset = CrackSegmentationDataset(
        root_dir=config.data_root,
        split='test',
        image_size=config.image_size,
        horizontal_flip=False,
        config=None,  # No augmentations for test set
    )

    # dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader
