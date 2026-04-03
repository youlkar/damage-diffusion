# training utilities for the DDPM model

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict
import json

from .metrics import MetricsTracker, compute_fid_score
from .visualization import visualize_samples, save_samples, plot_training_curves
from models.diffusion import EMAModel
from models.latent_diffusion import LatentDiffusionModel


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        config,
        device: str = 'cuda',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Initialize EMA if enabled
        self.ema = EMAModel(self.model, decay=config.ema_decay) if config.use_ema else None

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(log_dir=config.log_dir)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Mixed precision scaler
        self.use_amp = config.mixed_precision in ('fp16', 'bf16') and 'cuda' in device
        self.scaler = GradScaler() if self.use_amp else None
        self.amp_dtype = torch.float16 if config.mixed_precision == 'fp16' else torch.bfloat16

        print(f"Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Mixed precision: {config.mixed_precision} (AMP: {self.use_amp})")

    def train_epoch(self) -> float:
        # train for one epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Check if using latent diffusion
        is_latent_diffusion = isinstance(self.model, LatentDiffusionModel)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    loss = self._compute_loss(images, masks, is_latent_diffusion)
                
                # Backward pass with AMP
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    if is_latent_diffusion:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                loss = self._compute_loss(images, masks, is_latent_diffusion)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if is_latent_diffusion:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_parameters(),
                            self.config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                
                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log to TensorBoard
            if self.global_step % self.config.log_every_steps == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _compute_loss(self, images, masks, is_latent_diffusion):
        """Compute training loss for one batch."""
        if is_latent_diffusion:
            # For pre-encoded latents, images/masks are already latents
            if len(images.shape) == 4 and images.shape[1] == 4:  # Already latents
                image_latents = images
                mask_latents = masks
            else:
                # Encode to latent space (fallback for non-cached data)
                with torch.no_grad():
                    image_latents = self.model.encode_images(images)
                    mask_latents = self.model.encode_masks(masks)

            # Sample timesteps
            batch_size = image_latents.shape[0]
            timesteps = torch.randint(
                0, self.model.scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()

            # Sample noise
            noise = torch.randn_like(image_latents)

            # Forward pass in latent space
            noise_pred = self.model(image_latents, mask_latents, timesteps, noise)

            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(noise_pred, noise)
        else:
            # PIXEL-SPACE DDPM: Standard forward pass
            noise_pred, noise, noisy_images = self.model(images, masks)

            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(noise_pred, noise)
        
        return loss

    @torch.no_grad()
    def validate(self) -> float:
        # validate the model
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Check if using latent diffusion
        is_latent_diffusion = isinstance(self.model, LatentDiffusionModel)

        for images, masks in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Use mixed precision for validation too
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    loss = self._compute_loss(images, masks, is_latent_diffusion)
            else:
                loss = self._compute_loss(images, masks, is_latent_diffusion)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 8) -> torch.Tensor:
        # generate samples for visualization
        self.model.eval()

        # Check if using latent diffusion
        is_latent_diffusion = isinstance(self.model, LatentDiffusionModel)

        # Get random masks from validation set
        images, masks = next(iter(self.val_loader))
        masks = masks[:num_samples].to(self.device)
        images = images[:num_samples].to(self.device)

        # Apply EMA if enabled (only for pixel-space DDPM)
        if self.ema is not None and not is_latent_diffusion:
            self.ema.apply_shadow(self.model)

        # Generate images
        if is_latent_diffusion:
            # LATENT DIFFUSION: Encode masks and sample
            mask_latents = self.model.encode_masks(masks)
            num_steps = getattr(self.config, 'num_inference_steps', 50)
            generated = self.model.sample(mask_latents, num_inference_steps=num_steps)
        else:
            # PIXEL-SPACE DDPM
            generated = self.model.generate(
                masks,
                num_inference_steps=self.config.num_inference_steps
            )

        # Restore original weights (only for pixel-space DDPM)
        if self.ema is not None and not is_latent_diffusion:
            self.ema.restore(self.model)

        return images, masks, generated

    def save_checkpoint(self, filename: str, is_best: bool = False):
        # save model checkpoint
        is_latent_diffusion = isinstance(self.model, LatentDiffusionModel)

        # For latent diffusion, only save U-Net (VAE is pre-trained and frozen)
        if is_latent_diffusion:
            model_state = self.model.unet.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        # load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        is_latent_diffusion = isinstance(self.model, LatentDiffusionModel)

        # Load model state
        if is_latent_diffusion:
            # Only load U-Net (VAE is pre-trained and frozen)
            self.model.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def train(self):
        # main training loop
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'='*60}\n")

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.lr_scheduler.step()

            # Log metrics
            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=self.optimizer.param_groups[0]['lr']
            )

            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)

            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.config.save_checkpoint_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', is_best=is_best)

            # Generate and save samples
            if (epoch + 1) % self.config.generate_samples_every_epochs == 0:
                print("Generating samples...")
                images, masks, generated = self.generate_samples(
                    num_samples=self.config.num_samples_to_generate
                )

                # Visualize
                fig = visualize_samples(
                    images, masks, generated,
                    num_samples=min(4, self.config.num_samples_to_generate),
                    save_path=f"{self.config.sample_dir}/samples_epoch_{epoch}.png"
                )

                # Log to TensorBoard
                self.writer.add_figure('samples/generated', fig, epoch)

            # Compute FID if enabled
            if self.config.compute_fid and (epoch + 1) % self.config.fid_every_epochs == 0:
                print("Computing FID score...")
                try:
                    fid_score = self.compute_fid()
                    self.metrics_tracker.update(fid_score=fid_score)
                    self.writer.add_scalar('metrics/fid', fid_score, epoch)
                    print(f"FID Score: {fid_score:.2f}")
                except Exception as e:
                    print(f"Failed to compute FID: {e}")

        # Save final checkpoint
        self.save_checkpoint('final_model.pt', is_best=False)

        # Save metrics
        self.metrics_tracker.save()

        # Plot training curves
        plot_training_curves(
            self.metrics_tracker.metrics['train_loss'],
            self.metrics_tracker.metrics['val_loss'],
            save_path=f"{self.config.log_dir}/training_curves.png"
        )

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        self.writer.close()

    @torch.no_grad()
    def compute_fid(self) -> float:
        # compute FID score on validation set
        self.model.eval()

        # Collect real images
        real_images = []
        masks_for_generation = []

        for images, masks in self.val_loader:
            real_images.append(images)
            masks_for_generation.append(masks)

            if len(real_images) * images.shape[0] >= self.config.num_fid_samples:
                break

        real_images = torch.cat(real_images, dim=0)[:self.config.num_fid_samples]
        masks_for_generation = torch.cat(masks_for_generation, dim=0)[:self.config.num_fid_samples]

        # Generate fake images
        generated_images = []
        batch_size = self.config.eval_batch_size

        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        for i in range(0, len(masks_for_generation), batch_size):
            batch_masks = masks_for_generation[i:i+batch_size].to(self.device)
            batch_generated = self.model.generate(
                batch_masks,
                num_inference_steps=self.config.num_inference_steps
            )
            generated_images.append(batch_generated.cpu())

        if self.ema is not None:
            self.ema.restore(self.model)

        generated_images = torch.cat(generated_images, dim=0)

        # Compute FID
        fid = compute_fid_score(real_images, generated_images, device=self.device)

        return fid
