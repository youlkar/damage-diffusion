"""
eval.py — Standalone evaluation script for DamageDiffusion.

Usage:
    python eval.py --checkpoint checkpoints/best_model.pt --data_root /path/to/data
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.diffusion import MaskConditionedDDPM
from data.dataset import get_dataloaders
from utils.metrics import (
    compute_fid_score,
    compute_iou,
    compute_pixel_accuracy,
    _build_inception_model,
)
from configs import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DamageDiffusion model")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_root",   type=str, required=True)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--steps",       type=int, default=50)
    parser.add_argument("--output_dir",  type=str, default="./eval_results")
    parser.add_argument("--split",       type=str, default="test", choices=["test", "val"])
    parser.add_argument("--skip_fid",    action="store_true")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = TrainingConfig()
    for key, value in checkpoint.get("config", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.device = device

    model = MaskConditionedDDPM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "ema_state_dict" in checkpoint:
        print("Using EMA weights")
        for name, param in model.named_parameters():
            if name in checkpoint["ema_state_dict"]:
                param.data = checkpoint["ema_state_dict"][name].to(device)

    trained_epochs = checkpoint.get("epoch", "unknown")
    print(f"Trained for {trained_epochs} epochs")

    config.data_root = args.data_root
    _, val_loader, test_loader = get_dataloaders(config)
    loader = test_loader if args.split == "test" else val_loader

    print(f"\nCollecting {args.num_samples} samples...")
    real_images_list, masks_list = [], []
    for images, masks in loader:
        real_images_list.append(images)
        masks_list.append(masks)
        if sum(x.shape[0] for x in real_images_list) >= args.num_samples:
            break

    real_images = torch.cat(real_images_list, dim=0)[:args.num_samples]
    masks       = torch.cat(masks_list,        dim=0)[:args.num_samples]

    print(f"\nGenerating images ({args.steps} steps)...")
    generated_list = []
    for i in tqdm(range(0, len(masks), args.batch_size), desc="Generating"):
        batch_masks = masks[i:i + args.batch_size].to(device)
        generated_list.append(model.generate(batch_masks, num_inference_steps=args.steps).cpu())
    generated_images = torch.cat(generated_list, dim=0)

    print("\nComputing metrics...")
    all_iou, all_acc = [], []
    for i in range(0, len(generated_images), args.batch_size):
        gen_batch  = generated_images[i:i + args.batch_size]
        mask_batch = masks[i:i + args.batch_size]
        gen_mask   = (gen_batch.mean(dim=1, keepdim=True) > 0).float()
        all_iou.append(compute_iou(gen_mask, mask_batch))
        all_acc.append(compute_pixel_accuracy(gen_mask, mask_batch))

    results = {
        "iou":            float(np.mean(all_iou)),
        "pixel_accuracy": float(np.mean(all_acc)),
        "fid":            None,
        "checkpoint":     args.checkpoint,
        "trained_epochs": trained_epochs,
        "num_samples":    len(generated_images),
        "split":          args.split,
        "inf_steps":      args.steps,
    }

    if not args.skip_fid:
        print("Computing FID...")
        inception = _build_inception_model(device)
        results["fid"] = compute_fid_score(real_images, generated_images, device=device, inception_model=inception)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Trained epochs: {trained_epochs}")
    print(f"  Split         : {args.split} ({len(generated_images)} samples)")
    print(f"  Inf. steps    : {args.steps}")
    print("-"*50)
    print(f"  IoU            : {results['iou']:.4f}  (higher is better)")
    print(f"  Pixel Accuracy : {results['pixel_accuracy']:.4f}  (higher is better)")
    if results["fid"] is not None:
        print(f"  FID            : {results['fid']:.2f}   (lower is better)")
    else:
        print("  FID            : skipped")
    print("="*50)

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
