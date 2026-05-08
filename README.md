# DamageDiffusion Backend - Training pipeline
Mask-conditioned diffusion (DDPM-style) for generating crack images from binary masks. Training is driven by train.py; colab_train.ipynb is the Google Colab workflow (Drive + local SSD + Hugging Face data

## What the model does
- Input: RGB image + binary crack mask (4 channels: 3 + 1).
- Output: Denoised RGB (noise prediction), using Hugging Face Diffusers (UNet2DModel + DDPMScheduler), wrapped in MaskConditionedDDPM (models/diffusion.py).
- Conditioning: Mask concatenated with the noisy image; optional mask dropout for classifier-free–style training (mask_dropout_prob in TrainingConfig).
- Extras: EMA, weighted loss on crack pixels (crack_loss_weight), optional KID/FID via utils/metrics.py.


## Backend repository layout - the files and their functions

train.py: CLI entry -  config preset, hardware detection, dataloaders, Trainer.train().

inference.py:	Load checkpoint + generate from one mask, a mask dir, or batch settings.

configs/train_config.py:	TrainingConfig (default), MediumTrainingConfig, FastTrainingConfig.

data/dataset.py:	CrackSegmentationDataset, get_dataloaders (90/10 train/val from train/, test/ untouched for hold-out).

models/diffusion.py:	MaskConditionedDDPM (4-ch in, 3-ch noise out).

utils/training.py:	Training loop, validation, checkpoints, sampling, metric hooks.

utils/metrics.py:	FID/KID and related evaluation helpers.

utils/visualization.py:	Sample grids, denormalization, saving figures.

requirements.txt:	Dependencies (used by Colab install cell when present).

colab_train.ipynb:	End-to-end Colab: Drive, clone, dataset, deps, copy to /content, train, TensorBoard, inference, batch synth.

## Dataset layout 
crack_segmentation_dataset/
  train/
    images/   # *.jpg / *.png
    masks/    # same filenames as images
  test/
    images/
    masks/

## Local/server training
From the backend directory, run the following directly instead of colab notebook by creating an env:

cd backend
python3 -m venv .venv
source .venv/bin/activate (or .venv\Scripts\activate.bat command on windows)
pip install --upgrade pip
pip install -r requirements.txt

python train.py --config fast    # shorter/smaller run (see train_config.py)
python train.py --config medium  # validation-oriented medium run

python train.py \
  --data_root /path/to/crack_segmentation_dataset \
  --batch_size 16 \
  --num_epochs 100 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs \
  --resume /path/to/checkpoint.pt


## Colab training workflow - colab_train.ipynb
###  Run the cells in the notebook step by step or all at once. Below is the process in detail. IMPORTANT: ENSURE ATLEAST 2-3GB OR MORE OF FREE SPACE IS AVAILABLE ON GOOGLE DRIVE BEFORE RUNNING ON COLAB.
1. Configure top cell in notebook - GITHUB_REPO, PROJECT_DIR (Google Drive path), NUM_EPOCHS (actual epochs are being directly used in train.py unless changed in training command in notebook)

2. Mount Drive and verify access.

3. GPU: Notebook may set an initial BATCH_SIZE; train.py still runs its own VRAM-based sizing training is launched.

4. Clone/update repo under PROJECT_DIR (notebook currently targets a main branch and can be adjusted as needed).

5. Download or reuse Hugging Face dataset under {PROJECT_DIR}/data/crack_segmentation_dataset

6.pip install -r backend/requirements.txt from the cloned repo (fallback packages are listed in the notebook if that fails).

7. Copy dataset from Drive -> /content/crack_dataset (ephemeral SSD for fast training and inference but creates new at each runtime).

8. Train with command:
cd {PROJECT_DIR}/backend
python train.py \
  --data_root /content/crack_dataset \
  --checkpoint_dir {PROJECT_DIR}/checkpoints_full \
  --log_dir {PROJECT_DIR}/logs_full \
  --num_workers 4


9. Run TensorBoard on {PROJECT_DIR}/logs or logs_full to monitor the training. Checkpoints are under checkpoints_full folder after training

10. Inference: Notebook cells call inference.py with --checkpoint, --mask_path or --mask_dir, --output_dir, --num_samples, --num_steps to generate samples on the test images of the dataset.
python inference.py \
  --checkpoint path/to/best_model.pt \
  --mask_path path/to/mask.jpg \
  --output_dir path/to/out \
  --num_samples 4 \
  --num_steps 50

## Note: Checkpoints store training config when saved. inference.py restores it from the checkpoint when present.


## Default configuratiins (under configs/train_config.py)
- Image size 128×128, diffusion 1000 steps (default), linear beta schedule.
- KID/FID: compute_metrics, metrics_every_epochs, num_metrics_samples.
- Mask / loss: crack_loss_weight, mask_dropout_prob.
- Augmentation: Default emphasizes horizontal flip; rotation/color jitter and related flags differ between default vs fast configs—see class definitions


## Generated images for each of the test images from inferencing are stored in backend/samples for each


# DamageDiffusion Frontend

A Gradio-based frontend for generating synthetic concrete crack images from binary masks using the DamageDiffusion model.

This frontend is built to run in Google Colab and provides an interactive browser interface for:
- loading a trained model checkpoint,
- uploading or drawing crack masks,
- generating synthetic crack images,
- visualizing training metrics such as loss, FID, KID, and learning rate.

## Overview

DamageDiffusion is a mask-conditioned diffusion model for controllable crack image synthesis.  
This frontend makes the inference workflow easier to use by exposing the model through a Gradio web interface instead of requiring direct backend execution.

The main workflow is:
1. Load a trained checkpoint
2. Upload or draw a binary crack mask
3. Generate one or more synthetic crack images
4. Inspect model training behavior through the metrics dashboard

## Features

- Checkpoint loading with selectable device (`cuda` or `cpu`)
- Mask input through image upload or drawing canvas
- Automatic mask preprocessing for inference
- Synthetic crack image generation using the backend diffusion pipeline
- Training metrics dashboard for loss, FID, KID, and learning rate visualization
- Google Colab-friendly setup with shareable Gradio interface

## Tech Stack

- Python
- Gradio
- PyTorch
- Diffusers
- TorchMetrics
- Matplotlib
- Google Colab / Google Drive integration

## Requirements

Install the required packages before running the notebook:

```bash
pip install pytorch-fid torchmetrics -q
pip install -U gradio huggingface_hub diffusers
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

Depending on your environment, you may also need:
- `google-api-python-client`
- `google-auth`
- `torchvision`
- `matplotlib`

## Running the Frontend

### Recommended: Google Colab

This frontend was developed to run in Google Colab.

1. Open `app.ipynb` in Google Colab
2. Enable a GPU runtime
3. Run the dependency installation cells
4. Clone the repository
5. Download or mount the trained checkpoint and metrics file
6. Run the notebook cells that launch the Gradio app
7. Open the public Gradio link generated by Colab

## Required Files

The frontend expects access to:

- A trained model checkpoint, for example:
  - `/content/final_model.pt`
  - or a Google Drive checkpoint path
- A metrics file for dashboard visualization, for example:
  - `/content/checkpoint_metrics.pt`

If your files are stored elsewhere, update the paths in the notebook accordingly.

## Interface

### Model Loading

The app includes a shared control section with:
- a checkpoint path textbox,
- a device selector (`cuda` or `cpu`),
- a button to load the model.

The model is loaded on demand so the interface remains responsive until inference is needed.

### Generate Tab

The **Generate** tab is the main part of the frontend.

Users can:
- upload a binary crack mask, or
- draw a custom mask directly in the sketchpad.

The mask is then:
- resized to `128 x 128`,
- converted to grayscale,
- binarized,
- passed to the backend inference pipeline.

Generation settings include:
- number of inference steps,
- number of output samples.

The generated images are displayed in a gallery inside the app.

### Training Metrics Tab

The **Training Metrics** tab loads a saved metrics file and displays training curves in a 2×2 plot layout.

The dashboard includes:
- training loss,
- validation loss,
- FID score,
- KID score,
- learning rate schedule.

This gives a quick visual summary of model convergence and training progress.

## Backend Integration

The frontend uses backend modules from the repository, including:
- `load_model`
- `generate_from_mask`

These are imported from the backend inference pipeline and used to run mask-conditioned generation from the Gradio app.

## Workflow

1. Launch the notebook in Google Colab
2. Load the trained checkpoint
3. Select `cuda` if a GPU is available
4. Upload or draw a crack mask
5. Choose the number of inference steps
6. Generate one or more samples
7. Open the Training Metrics tab to inspect convergence plots

## Notes

- This frontend is intended for inference and visualization, not full training.
- A GPU is strongly recommended for faster generation.
- If `cuda` is selected but unavailable, the app falls back to CPU.
- The notebook includes some additional development utilities, but the main showcased frontend workflow is generation and metrics visualization.


## Acknowledgments

This frontend is part of the DamageDiffusion project, which focuses on controllable synthetic concrete crack generation using a mask-conditioned diffusion model.
