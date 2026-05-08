# Mask-Conditioned Denoising Diffusion Probabilistic Model (DDPM).
# Implements a U-Net based diffusion model with binary mask conditioning.
# Uses DDIM scheduler for fast, production-grade inference.


import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from typing import Optional, Tuple


class MaskConditionedDDPM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.image_size = config.image_size

        # init U-Net model with mask conditioning
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,  # 3 rgb + 1 mask
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
        )

        # Initialize noise scheduler for training (adding noise)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
        )

        # Initialize DDIM scheduler for fast inference
        # DDIM is non-Markovian and can skip timesteps efficiently
        # Provides 10x speedup (50 steps vs 500) with same quality
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
        )

        # For device compatibility
        self.device = config.device

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]

        # sample random noise if not available
        if noise is None:
            noise = torch.randn_like(images)

        # sample random timesteps if not available
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=images.device,
            ).long()

        # add noise to images (forward diffusion)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        # concatenate noisy image with mask (conditioning)
        model_input = torch.cat([noisy_images, masks], dim=1)

        # predict noise
        noise_pred = self.model(model_input, timesteps).sample

        return noise_pred, noise, noisy_images

    @torch.no_grad()
    def generate(
        self,
        masks: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate images from masks using DDIM sampling.

        DDIM (Denoising Diffusion Implicit Models) provides fast, deterministic sampling:
        - 50 steps achieves quality equivalent to DDPM's 500 steps
        - 10x faster inference (critical for production/frontend use)
        - Non-Markovian process allows safe timestep skipping

        Args:
            masks: Binary masks (B, 1, H, W)
            num_inference_steps: Number of denoising steps (50 recommended)
            generator: Random generator for reproducibility

        Returns:
            Generated images (B, 3, H, W) in range [-1, 1]
        """
        batch_size = masks.shape[0]

        # Start from random noise
        image = torch.randn(
            (batch_size, 3, self.image_size, self.image_size),
            generator=generator,
            device=masks.device,
        )

        # Set timesteps for DDIM scheduler
        # DDIM automatically creates optimal timestep spacing
        # Note: device parameter may not be supported in older diffusers versions
        try:
            self.inference_scheduler.set_timesteps(num_inference_steps, device=masks.device)
        except TypeError:
            # Fallback for older diffusers versions without device parameter
            self.inference_scheduler.set_timesteps(num_inference_steps)
            # Move timesteps to correct device
            self.inference_scheduler.timesteps = self.inference_scheduler.timesteps.to(masks.device)

        # DDIM denoising loop
        for t in self.inference_scheduler.timesteps:
            # Concatenate current noisy image with mask (conditioning)
            model_input = torch.cat([image, masks], dim=1)

            # Predict noise residual
            # Ensure timestep is a tensor on the correct device with correct dtype
            if isinstance(t, torch.Tensor):
                timestep = t.unsqueeze(0).expand(batch_size).to(masks.device)
            else:
                timestep = torch.tensor([t] * batch_size, device=masks.device, dtype=torch.long)

            noise_pred = self.model(model_input, timestep).sample

            # DDIM step: compute previous noisy sample
            # Uses deterministic reverse process (eta=0 by default)
            # Ensure t is properly handled (int or tensor)
            t_value = t.item() if isinstance(t, torch.Tensor) else t
            image = self.inference_scheduler.step(noise_pred, t_value, image).prev_sample

        return image

    @torch.no_grad()
    def generate_cfg(
        self,
        masks: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # Classifier-free guidance (CFG) for mask-conditioned image generation.

        batch_size = masks.shape[0]

        image = torch.randn(
            (batch_size, 3, self.image_size, self.image_size),
            generator=generator,
            device=masks.device,
        )

        try:
            self.inference_scheduler.set_timesteps(num_inference_steps, device=masks.device)
        except TypeError:
            self.inference_scheduler.set_timesteps(num_inference_steps)
            self.inference_scheduler.timesteps = self.inference_scheduler.timesteps.to(masks.device)

        # zero mask used for unconditional pass
        uncond_masks = torch.zeros_like(masks)

        for t in self.inference_scheduler.timesteps:
            if isinstance(t, torch.Tensor):
                timestep = t.unsqueeze(0).expand(batch_size).to(masks.device)
            else:
                timestep = torch.tensor([t] * batch_size, device=masks.device, dtype=torch.long)

            # conditioned pass — real mask
            model_input_cond = torch.cat([image, masks], dim=1)
            noise_pred_cond = self.model(model_input_cond, timestep).sample

            # unconditioned pass — zero mask
            model_input_uncond = torch.cat([image, uncond_masks], dim=1)
            noise_pred_uncond = self.model(model_input_uncond, timestep).sample

            # CFG combination: amplify the mask-conditioned direction
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            t_value = t.item() if isinstance(t, torch.Tensor) else t
            image = self.inference_scheduler.step(noise_pred, t_value, image).prev_sample

        return image

    def save_pretrained(self, save_directory: str):
        """Save model and schedulers for later use."""
        self.model.save_pretrained(f"{save_directory}/unet")
        self.noise_scheduler.save_pretrained(f"{save_directory}/scheduler")
        self.inference_scheduler.save_pretrained(f"{save_directory}/inference_scheduler")

    def load_pretrained(self, save_directory: str):
        """Load model and schedulers from saved directory."""
        self.model = UNet2DModel.from_pretrained(f"{save_directory}/unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(f"{save_directory}/scheduler")
        self.inference_scheduler = DDIMScheduler.from_pretrained(f"{save_directory}/inference_scheduler")


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data = self.original[name]
        self.original = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):

        self.shadow = state_dict
