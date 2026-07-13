#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: QJPEG down-sampling of the grayscale Border Collie at patch_size=256.

Treats the whole 256x256 grayscale image as a SINGLE patch, i.e. one 16-qubit
QJPEG circuit (2 * log2(256) = 16 qubits) -- the "whole-image" setting closest
to the original paper, with no tiling. The image is down-sampled at several
compression levels and compared with classical box-averaging.

Run:
    python demo_QJPEG_dog256.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.QJEPG import QJPEG

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / 'img'
OUT_DIR = HERE / 'output'


def box_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    p = img.shape[0] // factor
    return img.reshape(p, factor, p, factor).mean(axis=(1, 3))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    size = 256
    img = np.asarray(
        Image.open(IMG_DIR / 'Border Collie.webp').convert('L').resize((size, size)),
        dtype=np.float64,
    ) / 255.0
    print(f'Input grayscale dog: {img.shape}  ({2 * int(np.log2(size))}-qubit whole-image patch)')

    qjpeg = QJPEG(patch_size=size, wrap_H_layer=True)

    results = {}
    for n_discard in (1, 2, 3):
        t = time.time()
        out = qjpeg(img, n_discard=n_discard)
        ref = box_downsample(img, 2 ** n_discard)
        print(f'n_discard={n_discard}: {img.shape} -> {out.shape}  '
              f'PSNR(vs box-avg)={psnr(out, ref):.1f} dB  ({time.time() - t:.1f}s)')
        results[n_discard] = out

    OUT_DIR.mkdir(exist_ok=True)
    n = 1 + len(results)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'input {size}x{size}')
    for ax, (nd, out) in zip(axes[1:], results.items()):
        ax.imshow(out, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'QJPEG n_discard={nd}\n{out.shape[0]}x{out.shape[1]}')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    path = OUT_DIR / 'demo_QJPEG_dog256.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'Saved comparison figure to {path}')


if __name__ == '__main__':
    main()
