#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Quantum JPEG (QJPEG) image down-sampling.

Loads a sample image, down-samples it with the QJPEG quantum circuit at several
compression levels, and saves a side-by-side comparison against classical
box-averaging (which QJPEG reproduces when the Hadamard layers are enabled).

Run:
    python demo_QJPEG.py
"""

import sys
from pathlib import Path

import numpy as np

# make `pyqpanda_alg` importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QJEPG import QJPEG

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / 'img'
OUT_DIR = HERE / 'output'


def load_gray(path: Path, size: int) -> np.ndarray:
    """Load an image as a size x size float32 grayscale array in [0, 1]."""
    img = Image.open(path).convert('L').resize((size, size))
    return (np.asarray(img, dtype=np.float32) / 255.0)


def synthetic_gray(size: int) -> np.ndarray:
    """A smooth synthetic image, used when Pillow / sample images are absent."""
    x = np.linspace(0, 3 * np.pi, size)
    xx, yy = np.meshgrid(x, x)
    img = (np.sin(xx) * np.cos(yy) + np.sin(xx / 2)) * 0.25 + 0.5
    return img.astype(np.float32)


def box_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    P = img.shape[0]
    p = P // factor
    return img.reshape(p, factor, p, factor).mean(axis=(1, 3))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    size = 128
    src = IMG_DIR / 'OriginQ logo.jpg'
    if _HAS_PIL and src.exists():
        img = load_gray(src, size)
        title = src.name
    else:
        img = synthetic_gray(size)
        title = 'synthetic'
    print(f'Input image: {title}  shape={img.shape}  range=[{img.min():.3f}, {img.max():.3f}]')

    # Down-sample the whole image as a single patch (most faithful to the paper).
    qjpeg = QJPEG(patch_size=size, wrap_H_layer=True)

    results = {}
    for n_discard in (1, 2, 3):
        out = qjpeg(img, n_discard=n_discard)
        ref = box_downsample(img, 2 ** n_discard)
        print(f'n_discard={n_discard}: {img.shape} -> {out.shape}  '
              f'PSNR(QJPEG vs box-avg)={psnr(out, ref):.1f} dB')
        results[n_discard] = out

    # Save a comparison figure if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        OUT_DIR.mkdir(exist_ok=True)
        n = 1 + len(results)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'input {img.shape[0]}x{img.shape[1]}')
        for ax, (nd, out) in zip(axes[1:], results.items()):
            ax.imshow(out, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'QJPEG n_discard={nd}\n{out.shape[0]}x{out.shape[1]}')
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        path = OUT_DIR / 'demo_QJPEG.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'Saved comparison figure to {path}')
    except ImportError:
        print('matplotlib not available; skipping figure.')


if __name__ == '__main__':
    main()
