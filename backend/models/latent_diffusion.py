# Latent Diffusion Model for mask-conditioned crack generation.

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from typing import Tuple, Optional


class LatentDiffusionModel(nn.Module):
    # latent Diffusion Model with mask conditioning.
    # steps: 
    # VAE Encoder -> 128x128x3 -> 16x16x4 (pre-trained, frozen)
    # DDPM U-Net -> denoising in 16x16x4 latent space
    # VAE Decoder -> 16x16x4 -> 128x128x3 (pre-trained, frozen)

    def __init__(
        self,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 256),
        num_train_timesteps: int = 100,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.latent_channels = latent_channels
        self.vae_scaling_factor = 0.18215  # Stable Diffusion scaling factor

        # Load pre-trained VAE (frozen)
        print(f"Loading pre-trained VAE: {vae_model_name}")
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)
        self.vae.eval()
        self.vae.requires_grad_(False)  # Freeze VAE
        self.vae.to(device)

        # Calculate latent size (128 / 8 = 16 for standard SD-VAE)
        self.latent_size = 16  # 8x downsampling factor

        # Input = 8 channels (4 image latent + 4 mask latent)
        # Output =  4 channels (predicted noise in image latent)
        self.unet = UNet2DModel(
            sample_size=self.latent_size,
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.unet.to(device)

        # DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

        print(f"Latent Diffusion Model initialized:")
        print(f"VAE: {vae_model_name} (frozen)")
        print(f"Latent space: {self.latent_size}x{self.latent_size}x{latent_channels}")
        print(f"U-Net channels: {block_out_channels}")
        print(f"Timesteps: {num_train_timesteps}")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # encode images to latent space using pre-trained VAE
        images = images.to(self.device)

        # VAE encode
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample()

        # scale latents (Stable Diffusion convention)
        latents = latents * self.vae_scaling_factor

        return latents

    @torch.no_grad()
    def encode_masks(self, masks: torch.Tensor) -> torch.Tensor:
        # encode binary masks to latent space
        masks = masks.to(self.device)

        # convert single-channel mask to 3-channel (repeat)
        masks_rgb = masks.repeat(1, 3, 1, 1)

        # normalize to [-1, 1] for VAE
        masks_rgb = masks_rgb * 2.0 - 1.0

        # VAE encode
        latent_dist = self.vae.encode(masks_rgb).latent_dist
        mask_latents = latent_dist.sample()

        # Scale latents
        mask_latents = mask_latents * self.vae_scaling_factor

        return mask_latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # decode latents to pixel space using pre-trained VAE
        latents = latents.to(self.device)

        # un-scale latents
        latents = latents / self.vae_scaling_factor

        # VAE decode
        images = self.vae.decode(latents).sample

        return images

    def forward(
        self,
        image_latents: torch.Tensor,
        mask_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # forward diffusion training step
        # sample noise if not provided
        if noise is None:
            noise = torch.randn_like(image_latents)

        # add noise to image latents (forward diffusion)
        noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)

        # concatenate noisy image latent with mask latent (conditioning)
        latent_model_input = torch.cat([noisy_latents, mask_latents], dim=1)

        # predict noise with U-Net
        noise_pred = self.unet(latent_model_input, timesteps).sample

        return noise_pred

    @torch.no_grad()
    def sample(
        self,
        mask_latents: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # generate images from masks using DDPM sampling
        batch_size = mask_latents.shape[0]

        # Start from random noise
        latents = torch.randn(
            (batch_size, self.latent_channels, self.latent_size, self.latent_size),
            generator=generator,
            device=self.device,
        )

        # Set scheduler for inference
        self.scheduler.set_timesteps(num_inference_steps)

        # denoising loop
        for t in self.scheduler.timesteps:
            # concatenate noisy latent with mask latent
            latent_model_input = torch.cat([latents, mask_latents], dim=1)

            # predict noise
            noise_pred = self.unet(
                latent_model_input,
                t.unsqueeze(0).expand(batch_size).to(self.device)
            ).sample

            # remove noise (reverse diffusion step)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to images
        images = self.decode_latents(latents)

        return images

    def get_trainable_parameters(self):
        # get only U-Net parameters (VAE is frozen)
        return self.unet.parameters()

    def get_model_size(self) -> dict:
        # get model size information
        unet_params = sum(p.numel() for p in self.unet.parameters())
        vae_params = sum(p.numel() for p in self.vae.parameters())

        return {
            "unet_params": unet_params,
            "vae_params": vae_params,
            "total_params": unet_params + vae_params,
            "trainable_params": unet_params,
        }
