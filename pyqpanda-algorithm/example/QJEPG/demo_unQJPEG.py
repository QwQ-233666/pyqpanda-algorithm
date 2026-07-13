#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: inverse Quantum JPEG (unQJPEG) image up-sampling / super-resolution.

Down-samples a sample image, then up-samples it back with the unQJPEG quantum
circuit (spectral zero-padding / sinc interpolation) and reports how well the
original resolution is recovered. Also demonstrates plain up-sampling of a
low-resolution image.

Run:
    python demo_unQJPEG.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QJEPG import QJPEG, unQJPEG

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / 'img'
OUT_DIR = HERE / 'output'


def load_gray(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert('L').resize((size, size))
    return (np.asarray(img, dtype=np.float32) / 255.0)


def synthetic_gray(size: int) -> np.ndarray:
    x = np.linspace(0, 3 * np.pi, size)
    xx, yy = np.meshgrid(x, x)
    img = (np.sin(xx) * np.cos(yy) + np.sin(xx / 2)) * 0.25 + 0.5
    return img.astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    size = 128
    half = size // 2
    src = IMG_DIR / 'Border Collie.webp'
    if _HAS_PIL and src.exists():
        img = load_gray(src, size)
        title = src.name
    else:
        img = synthetic_gray(size)
        title = 'synthetic'
    print(f'Input image: {title}  shape={img.shape}')

    # 1) low-res image obtained by QJPEG down-sampling (128 -> 64)
    low = QJPEG(patch_size=size, wrap_H_layer=True)(img, n_discard=1)
    print(f'Down-sampled with QJPEG: {img.shape} -> {low.shape}')

    # 2) up-sample it back to full resolution (64 -> 128) with unQJPEG
    recon = unQJPEG()(low, n_append=1, patch_size=half)
    recon = np.clip(recon, 0.0, 1.0)
    print(f'Up-sampled with unQJPEG: {low.shape} -> {recon.shape}  '
          f'PSNR(recon vs original)={psnr(recon, img):.1f} dB')

    # baseline: nearest-neighbour upscaling of the same low-res image
    nn = np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)
    print(f'(baseline nearest-neighbour PSNR={psnr(nn, img):.1f} dB)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        OUT_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, im, t in zip(
            axes,
            [img, low, recon, nn],
            [f'original {size}x{size}', f'QJPEG low {half}x{half}',
             f'unQJPEG {size}x{size}', f'nearest {size}x{size}'],
        ):
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
            ax.set_title(t)
            ax.axis('off')
        fig.tight_layout()
        path = OUT_DIR / 'demo_unQJPEG.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'Saved comparison figure to {path}')
    except ImportError:
        print('matplotlib not available; skipping figure.')


if __name__ == '__main__':
    main()
