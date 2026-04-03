# Mask-Conditioned Denoising Diffusion Probabilistic Model (DDPM).
# Implements a U-Net based diffusion model with binary mask conditioning.


import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler
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

        # Initialize noise scheduler
        self.noise_scheduler = DDPMScheduler(
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

        batch_size = masks.shape[0]

        # Start from random noise
        image = torch.randn(
            (batch_size, 3, self.image_size, self.image_size),
            generator=generator,
            device=masks.device,
        )

        # set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # denoising loop
        for t in self.noise_scheduler.timesteps:
            # concatenate current noisy image with mask
            model_input = torch.cat([image, masks], dim=1)

            # predict noise
            timestep = torch.tensor([t] * batch_size, device=masks.device)
            noise_pred = self.model(model_input, timestep).sample

            # compute previous noisy sample
            image = self.noise_scheduler.step(noise_pred, t, image).prev_sample

        return image

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(f"{save_directory}/unet")
        self.noise_scheduler.save_pretrained(f"{save_directory}/scheduler")

    def load_pretrained(self, save_directory: str):
        self.model = UNet2DModel.from_pretrained(f"{save_directory}/unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(f"{save_directory}/scheduler")


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
