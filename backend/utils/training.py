# training utilities for the DDPM model

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict, Tuple
import json

from .metrics import MetricsTracker, compute_fid_score, compute_fid_kid_scores
from .visualization import visualize_samples, save_samples, plot_training_curves, plot_metrics
from models.diffusion import EMAModel


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

        # Mixed precision training
        self.use_amp = config.mixed_precision in ('fp16', 'bf16') and 'cuda' in device
        self.scaler = GradScaler() if self.use_amp and config.mixed_precision == 'fp16' else None
        self.amp_dtype = torch.float16 if config.mixed_precision == 'fp16' else torch.bfloat16

        print(f"Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Mixed precision: {config.mixed_precision} (AMP: {self.use_amp})")

    def train_epoch(self) -> float:
        # train for one epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move to device with channels-last format for GPU efficiency
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    noise_pred, noise, noisy_images = self.model(images, masks)
                    loss = F.mse_loss(noise_pred, noise)

                # Backward pass with AMP
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                noise_pred, noise, noisy_images = self.model(images, masks)
                loss = F.mse_loss(noise_pred, noise)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
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

    @torch.no_grad()
    def validate(self) -> float:
        # validate the model
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for images, masks in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    noise_pred, noise, noisy_images = self.model(images, masks)
                    loss = F.mse_loss(noise_pred, noise)
            else:
                noise_pred, noise, noisy_images = self.model(images, masks)
                loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 8) -> torch.Tensor:
        # generate samples for visualization
        self.model.eval()

        # clear cuda cache before generating samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get random masks from validation set
        images, masks = next(iter(self.val_loader))
        masks = masks[:num_samples].to(self.device)
        images = images[:num_samples].to(self.device)

        # Apply EMA if enabled
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        # Generate images
        generated = self.model.generate(
            masks,
            num_inference_steps=self.config.num_inference_steps
        )

        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)

        return images, masks, generated

    def save_checkpoint(self, filename: str, is_best: bool = False):
        # save model checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
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

            # Compute FID and KID if enabled
            if self.config.compute_metrics and epoch % self.config.metrics_every_epochs == 0:
                print("Computing metrics...")
                try:
                    fid_score, kid_score = self.compute_fid_kid()
                    self.metrics_tracker.update(fid_score=fid_score, kid_score=kid_score)
                    self.writer.add_scalar('metrics/fid', fid_score, epoch)
                    self.writer.add_scalar('metrics/kid', kid_score, epoch)
                    print(f"FID Score: {fid_score:.2f}")
                    print(f"KID Score: {kid_score:.2f}")
                except Exception as e:
                    print(f"Failed to compute FID or KID: {e}")

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
        
        plot_metrics(
            self.metrics_tracker.metrics['fid_score'],
            'FID',
            save_path=f"{self.config.log_dir}/fid_scores.png"
        )
        
        plot_metrics(
            self.metrics_tracker.metrics['kid_score'],
            'KID',
            save_path=f"{self.config.log_dir}/kid_scores.png"
        )

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        self.writer.close()

    # Possibly deprecated
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

            if len(real_images) * images.shape[0] >= self.config.num_metrics_samples:
                break

        real_images = torch.cat(real_images, dim=0)[:self.config.num_metrics_samples]
        masks_for_generation = torch.cat(masks_for_generation, dim=0)[:self.config.num_metrics_samples]

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

    # Computing FID and KID metrics together to avoid regenerating for each
    @torch.no_grad()
    def compute_fid_kid(self) -> Tuple[float, float]:
        # compute scores on validation set
        self.model.eval()

        # Collect real images
        real_images = []
        masks_for_generation = []

        for images, masks in self.val_loader:
            real_images.append(images)
            masks_for_generation.append(masks)

            if len(real_images) * images.shape[0] >= self.config.num_metrics_samples:
                break

        real_images = torch.cat(real_images, dim=0)[:self.config.num_metrics_samples]
        masks_for_generation = torch.cat(masks_for_generation, dim=0)[:self.config.num_metrics_samples]

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

        # Compute scores
        fid, kid = compute_fid_kid_scores(real_images, generated_images)

        return fid, kid