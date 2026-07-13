#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: QJPEG / unQJPEG on multi-channel images (RGB and RGB+IR).

The QJPEG pipeline processes every image channel independently, so it works
out-of-the-box for 3-channel RGB and 4-channel RGB+IR (near-infrared) images.

This demo uses an aligned RGB / NIR scene from the EPFL RGB-NIR Scene Dataset
(country_0000): vegetation reflects strongly in the near-infrared, so the IR
channel carries clearly different information than the visible channels -- a good
illustration that the 4th (IR) channel is compressed and reconstructed just like
the colour channels.

Outputs (written to ./output):
    demo_RGB.png     -- QJPEG down-sampling + unQJPEG up-sampling of the RGB image
    demo_RGB_IR.png  -- same for the RGB+IR image, showing colour and IR views

Run:
    python demo_QJPEG_RGB_IR.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QJEPG import QJPEG, unQJPEG

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / 'img'
OUT_DIR = HERE / 'output'


def load_rgb(path: Path, size: int) -> np.ndarray:
    """-> (3, size, size) float64 in [0, 1]."""
    im = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64).transpose(2, 0, 1) / 255.0


def load_gray(path: Path, size: int) -> np.ndarray:
    """-> (size, size) float64 in [0, 1]."""
    im = Image.open(path).convert('L').resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    return np.clip(img.transpose(1, 2, 0), 0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    size = 128
    half = size // 2
    OUT_DIR.mkdir(exist_ok=True)

    rgb = load_rgb(IMG_DIR / 'country_rgb.png', size)          # (3, H, W)
    nir = load_gray(IMG_DIR / 'country_nir.png', size)         # (H, W)
    rgb_ir = np.concatenate([rgb, nir[None]], axis=0)          # (4, H, W)
    print(f'RGB image   : {rgb.shape}')
    print(f'RGB+IR image: {rgb_ir.shape}  (channels: R, G, B, IR)')

    qjpeg = QJPEG(patch_size=size, wrap_H_layer=True)
    unqjpeg = unQJPEG()

    # ---- RGB: down-sample x2, x4 and up-sample round-trip ----
    rgb_d1 = np.clip(qjpeg(rgb, n_discard=1), 0, 1)            # (3, 64, 64)
    rgb_d2 = np.clip(qjpeg(rgb, n_discard=2), 0, 1)           # (3, 32, 32)
    rgb_up = unqjpeg(rgb_d1, n_append=1, patch_size=half)      # (3, 128, 128)
    print(f'RGB   : {rgb.shape[1:]} -> QJPEG {rgb_d1.shape[1:]} / {rgb_d2.shape[1:]}; '
          f'unQJPEG {rgb_up.shape[1:]}  PSNR(up vs orig)={psnr(np.clip(rgb_up,0,1), rgb):.1f} dB')

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, im, t in zip(
        ax,
        [rgb, rgb_d1, rgb_d2, np.clip(rgb_up, 0, 1)],
        [f'RGB input {size}x{size}', 'QJPEG x2  64x64',
         'QJPEG x4  32x32', f'unQJPEG up  {size}x{size}'],
    ):
        a.imshow(chw_to_hwc(im)); a.set_title(t); a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'demo_RGB.png', dpi=120, bbox_inches='tight')
    print(f'Saved {OUT_DIR / "demo_RGB.png"}')

    # ---- RGB+IR: process all 4 channels, view colour and IR separately ----
    ir_d1 = np.clip(qjpeg(rgb_ir, n_discard=1), 0, 1)          # (4, 64, 64)
    ir_up = unqjpeg(ir_d1, n_append=1, patch_size=half)        # (4, 128, 128)
    ir_up = np.clip(ir_up, 0, 1)
    print(f'RGB+IR: {rgb_ir.shape[1:]} -> QJPEG {ir_d1.shape[1:]}; unQJPEG {ir_up.shape[1:]}  '
          f'PSNR RGB={psnr(ir_up[:3], rgb):.1f} dB  PSNR IR={psnr(ir_up[3], nir):.1f} dB')

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    # top row: colour (R,G,B) view
    ax[0, 0].imshow(chw_to_hwc(rgb_ir[:3])); ax[0, 0].set_title(f'RGB view  {size}x{size}')
    ax[0, 1].imshow(chw_to_hwc(ir_d1[:3])); ax[0, 1].set_title('QJPEG x2  64x64')
    ax[0, 2].imshow(chw_to_hwc(ir_up[:3])); ax[0, 2].set_title(f'unQJPEG up  {size}x{size}')
    # bottom row: IR channel view
    ax[1, 0].imshow(rgb_ir[3], cmap='inferno', vmin=0, vmax=1); ax[1, 0].set_title(f'IR view  {size}x{size}')
    ax[1, 1].imshow(ir_d1[3], cmap='inferno', vmin=0, vmax=1); ax[1, 1].set_title('QJPEG x2  64x64')
    ax[1, 2].imshow(ir_up[3], cmap='inferno', vmin=0, vmax=1); ax[1, 2].set_title(f'unQJPEG up  {size}x{size}')
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'demo_RGB_IR.png', dpi=120, bbox_inches='tight')
    print(f'Saved {OUT_DIR / "demo_RGB_IR.png"}')


if __name__ == '__main__':
    main()
