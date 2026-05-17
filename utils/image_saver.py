"""Image saving utility for SRVAR training visualization.

Ported from myvaex/utils/image_saver.py and adapted for the SR task:
the comparison grid is 3 columns (LR_upsampled | HR_pred | HR_gt) instead of 2.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (clamped)."""
    return tensor.add(1).mul_(0.5).clamp_(0, 1)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    img_np = tensor.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    if img_np.shape[2] == 1:
        return Image.fromarray(img_np.squeeze(2), mode='L')
    return Image.fromarray(img_np, mode='RGB')


def _resize_to(tensor: torch.Tensor, target_hw: torch.Size) -> torch.Tensor:
    """Resize `[N,C,h,w]` -> `[N,C,target_h,target_w]` using bicubic."""
    if tensor.shape[-2:] == target_hw:
        return tensor
    return F.interpolate(tensor, size=tuple(target_hw), mode='bicubic', align_corners=False).clamp_(0, 1)


def save_reconstruction_comparison(
    lr: torch.Tensor,
    hr_pred: torch.Tensor,
    hr_gt: torch.Tensor,
    save_dir: str,
    ep: int,
    it: int,
    max_samples: int = 4,
) -> str:
    """Save a 3-column comparison: `LR_upsampled | HR_pred | HR_gt`.

    All tensors should be `[B, C, H, W]` in [-1, 1]. LR is bicubic-upsampled to
    HR's spatial size only for visualization (no metric effect).
    """
    os.makedirs(save_dir, exist_ok=True)

    lr_denorm = denormalize_image(lr.clone())
    pred_denorm = denormalize_image(hr_pred.clone())
    gt_denorm = denormalize_image(hr_gt.clone())

    target_hw = gt_denorm.shape[-2:]
    lr_up = _resize_to(lr_denorm, target_hw)

    batch_size = gt_denorm.shape[0]
    num_samples = min(batch_size, max_samples)

    tiles: List[torch.Tensor] = []
    for i in range(num_samples):
        tiles.append(lr_up[i])
        tiles.append(pred_denorm[i])
        tiles.append(gt_denorm[i])
    comparison_tensor = torch.stack(tiles, dim=0)

    grid = torchvision.utils.make_grid(
        comparison_tensor,
        nrow=3,
        padding=2,
        pad_value=1.0,
    )
    grid_pil = tensor_to_pil_image(grid)
    filename = f"ep{ep:04d}_it{it:06d}_comparison.png"
    filepath = os.path.join(save_dir, filename)
    grid_pil.save(filepath)
    return filepath


def save_reconstruction_run_metadata(
    save_dir: str,
    args_state: Dict[str, Any],
    stage_name: str,
    frequency_description: str,
    max_samples: int,
    filename_pattern: str = "ep{epoch:04d}_it{iter:06d}_comparison.png",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, "run_metadata.json")
    payload = {
        "stage_name": stage_name,
        "save_dir": save_dir,
        "filename_pattern": filename_pattern,
        "frequency_description": frequency_description,
        "comparison_layout": "3 columns per row: LR_upsampled | HR_pred | HR_gt",
        "max_samples_per_image": int(max_samples),
        "postprocess": [
            "denormalize LR/HR_pred/HR_gt from [-1, 1] to [0, 1]",
            "clamp values to [0, 1]",
            "bicubic upsample LR to HR resolution (visualization only)",
            "stack triples into a grid with white padding",
            "convert to uint8 PNG without extra filtering",
        ],
        "args": args_state,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)
    return metadata_path


def compute_psnr_ssim(
    hr_pred: torch.Tensor,
    hr_gt: torch.Tensor,
) -> Dict[str, float]:
    """Compute PSNR/SSIM as in myvaex: skimage on RGB [0, 1], data_range=1.0.

    Args:
        hr_pred, hr_gt: `[B, C, H, W]` in [-1, 1].

    Returns:
        Dict with keys `psnr_mean`, `ssim_mean`, and per-sample lists.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio as _psnr
        from skimage.metrics import structural_similarity as _ssim
    except ImportError as e:
        raise ImportError(
            "compute_psnr_ssim requires scikit-image (`pip install scikit-image`)."
        ) from e

    pred = (hr_pred + 1.0) * 0.5
    gt = (hr_gt + 1.0) * 0.5
    pred = pred.clamp(0, 1).detach().cpu().numpy()
    gt = gt.clamp(0, 1).detach().cpu().numpy()

    psnrs: List[float] = []
    ssims: List[float] = []
    for i in range(pred.shape[0]):
        p = np.transpose(pred[i], (1, 2, 0))     # HWC
        g = np.transpose(gt[i], (1, 2, 0))
        psnrs.append(float(_psnr(g, p, data_range=1.0)))
        # channel_axis=2 for HWC; SSIM requires same shape, default Gaussian window.
        ssims.append(float(_ssim(g, p, data_range=1.0, channel_axis=2)))

    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "psnr_list": psnrs,
        "ssim_list": ssims,
    }
