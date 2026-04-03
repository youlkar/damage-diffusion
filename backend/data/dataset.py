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
from tqdm import tqdm



class CrackSegmentationDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 128,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        horizontal_flip: bool = True,
    ):
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip and split == 'train'


        if split in ['train', 'val']:
            self.image_dir = self.root_dir / 'train' / 'images'
            self.mask_dir = self.root_dir / 'train' / 'masks'
        else:  # test
            self.image_dir = self.root_dir / 'test' / 'images'
            self.mask_dir = self.root_dir / 'test' / 'masks'

        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])

        self.valid_pairs = self._validate_pairs()

        print(f"Loaded {len(self.valid_pairs)} {split} samples from {self.image_dir}")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # Converts to [0, 1]
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform

        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),  # Converts to [0, 1]
            ])
        else:
            self.mask_transform = mask_transform

    def _validate_pairs(self) -> List[str]:
        """Validate that each image has a corresponding mask."""
        valid_pairs = []
        for img_file in self.image_files:
            mask_file = img_file  # Masks have the same filename as images
            if os.path.exists(self.mask_dir / mask_file):
                valid_pairs.append(img_file)
            else:
                print(f"Warning: No mask found for {img_file}")
        return valid_pairs

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.valid_pairs[idx]

        # Load image and mask
        image_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        if self.horizontal_flip and torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        return image, mask

    @staticmethod
    def create_train_val_split(
        root_dir: str,
        train_ratio: float = 0.9,
        image_size: int = 128,
        random_seed: int = 42,
        horizontal_flip: bool = True,
    ) -> Tuple[Dataset, Dataset]:
  
        full_dataset = CrackSegmentationDataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            horizontal_flip=horizontal_flip,
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
        )


        val_dataset.dataset = val_dataset_no_aug
        print(f"Split dataset: {train_size} train, {val_size} validation samples")

        return train_dataset, val_dataset

    def get_sample_batch(self, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of samples for visualization."""
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        images, masks = [], []
        for idx in indices:
            img, mask = self[idx]
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)


def get_dataloaders(config):
    # Check if using latent diffusion
    use_latent = hasattr(config, 'use_latent_diffusion') and config.use_latent_diffusion
    
    if use_latent:
        # For latent diffusion, use pre-encoded latent datasets
        train_dataset, val_dataset = create_latent_datasets(config)
    else:
        # Create standard datasets
        train_dataset, val_dataset = CrackSegmentationDataset.create_train_val_split(
            root_dir=config.data_root,
            train_ratio=config.train_val_split,
            image_size=config.image_size,
            random_seed=config.random_seed,
            horizontal_flip=config.horizontal_flip,
        )

    # Apply subset if configured (for fast training)
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
    )

    # Create dataloaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=4 if config.num_workers > 0 else 2,
        drop_last=True,  # Consistent batch sizes for AMP
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=4 if config.num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader


class LatentDataset(Dataset):
    """Pre-encoded latent dataset for faster LDM training."""
    
    def __init__(self, image_latents, mask_latents, indices=None):
        self.image_latents = image_latents
        self.mask_latents = mask_latents
        self.indices = indices if indices is not None else list(range(len(image_latents)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.image_latents[actual_idx], self.mask_latents[actual_idx]


def create_latent_datasets(config):
    """Create pre-encoded latent datasets for LDM training."""
    import os
    from pathlib import Path
    
    # Check for cached latents
    cache_dir = Path(config.data_root) / "latent_cache"
    cache_file = cache_dir / f"latents_{config.image_size}.pt"
    
    if cache_file.exists():
        print(f"Loading cached latents from {cache_file}")
        cached_data = torch.load(cache_file, map_location='cpu')
        image_latents = cached_data['image_latents']
        mask_latents = cached_data['mask_latents']
        train_indices = cached_data['train_indices']
        val_indices = cached_data['val_indices']
    else:
        print("Pre-encoding images to latent space (this may take 10-15 minutes, but only runs once)...")
        
        # Create standard dataset to encode
        full_dataset = CrackSegmentationDataset(
            root_dir=config.data_root,
            split='train',
            image_size=config.image_size,
            horizontal_flip=False,  # No augmentation for encoding
        )
        
        # Create temporary model for encoding
        from models.latent_diffusion import LatentDiffusionModel
        temp_model = LatentDiffusionModel(
            vae_model_name=config.vae_model,
            device=config.device
        )
        
        # Encode all data
        image_latents = []
        mask_latents = []
        
        # Use larger batch size for encoding
        encode_loader = DataLoader(
            full_dataset, 
            batch_size=min(32, len(full_dataset)), 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        with torch.no_grad():
            for images, masks in tqdm(encode_loader, desc="Encoding to latent space"):
                img_lat = temp_model.encode_images(images.to(config.device))
                mask_lat = temp_model.encode_masks(masks.to(config.device))
                
                image_latents.append(img_lat.cpu())
                mask_latents.append(mask_lat.cpu())
        
        image_latents = torch.cat(image_latents, dim=0)
        mask_latents = torch.cat(mask_latents, dim=0)
        
        # Create train/val split
        total_size = len(image_latents)
        train_size = int(config.train_val_split * total_size)
        
        generator = torch.Generator().manual_seed(config.random_seed)
        indices = torch.randperm(total_size, generator=generator)
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()
        
        # Cache the results
        cache_dir.mkdir(exist_ok=True)
        torch.save({
            'image_latents': image_latents,
            'mask_latents': mask_latents,
            'train_indices': train_indices,
            'val_indices': val_indices,
        }, cache_file)
        print(f"Cached latents to {cache_file}")
        
        # Clean up temporary model
        del temp_model
        torch.cuda.empty_cache()
    
    # Apply subset if configured
    if hasattr(config, 'subset_ratio') and config.subset_ratio < 1.0:
        original_size = len(train_indices)
        subset_size = int(original_size * config.subset_ratio)
        train_indices = train_indices[:subset_size]
        print(f"Using {config.subset_ratio*100:.0f}% data subset: {subset_size:,}/{original_size:,} samples")
    
    # Create datasets
    train_dataset = LatentDataset(image_latents, mask_latents, train_indices)
    val_dataset = LatentDataset(image_latents, mask_latents, val_indices)
    
    print(f"Latent datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset
