# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
from numpy import ndarray

from pyqpanda3.core import CPUQVM, QProg, Encode, H, CNOT

from ..plugin import QFT


class unQJPEG:
    """
    Inverse Quantum JPEG (unQJPEG) image up-sampling / super-resolution.

    unQJPEG is the inverse of :class:`~pyqpanda_alg.QJEPG.QJPEG`, originally
    proposed within this library. Where QJPEG *discards* the high spatial-frequency
    qubits to down-sample an image, unQJPEG *re-introduces* them as vacuum (zeros),
    i.e. it zero-pads the spatial-frequency spectrum and transforms back, which is
    the quantum analogue of sinc / Fourier interpolation.

    Each ``patch_size`` x ``patch_size`` patch is encoded into ``2 * log2(patch_size)``
    qubits (QPIE amplitude encoding). For each of the two image axes the quantum
    Fourier transform maps the register to the frequency domain; ``n_append`` fresh
    ``|0>`` qubits are inserted between the low- and high-frequency components (the
    high-frequency band is padded with zeros); an inverse QFT of the enlarged
    register returns to the spatial domain. Reading the probabilities yields an
    up-sampled patch of side ``patch_size * 2 ** n_append``.

    The sign qubit (most-significant frequency bit) of each axis is duplicated onto
    the appended qubits with ``CNOT`` gates so that the negative-frequency band is
    mapped to the top of the enlarged register, preserving the Hermitian frequency
    layout required for a real, interpolating reconstruction.

    Parameters
        wrap_H_layer : ``bool``, optional\n
            Kept for interface parity with :class:`QJPEG`. Unlike the down-sampling
            direction, the spectral (zero-padding) inverse must **not** be wrapped
            with Hadamard layers -- doing so destroys the reconstruction -- so this
            defaults to ``False`` and enabling it is only for experimentation.
        shots : ``int``, optional\n
            If given, the output probabilities are sampled with this many
            measurement shots (emulating finite-sampling noise). If ``None``
            (Default), the exact probabilities are used.
        seed : ``int``, optional\n
            Random seed used when ``shots`` is set (Default: None).

    References
        [1] Simone Roncallo, Lorenzo Maccone, Chiara Macchiavello. "Quantum JPEG".
        AVS Quantum Science 5, 043803 (2023). arXiv:2306.09323.

    >>> import numpy as np
    >>> from pyqpanda_alg.QJEPG import unQJPEG
    >>> img = np.random.rand(8, 8).astype(np.float64)
    >>> unqjpeg = unQJPEG()
    >>> out = unqjpeg(img, n_append=1, patch_size=8)     # 8x8 -> 16x16
    >>> print(out.shape)
    (16, 16)
    """

    def __init__(self, wrap_H_layer: bool = False, shots: int = None, seed: int = None):
        self.wrap_H_layer = wrap_H_layer
        self.shots = shots
        self.seed = seed
        # per-call state
        self.patch_size = None
        self.n_qubit = None
        self._grid = None      # (n_patch_H, n_patch_W)
        self._scale = 1        # up-sampling factor per axis (2 ** n_append)

    def __call__(self, img: ndarray, n_append: int = 1, patch_size: int = 128) -> ndarray:
        assert isinstance(img, ndarray)
        assert img.dtype in [np.float32, np.float64], \
            'image must be a float array in [0, 1]'
        assert 0 <= img.min() and img.max() <= 1.0, 'image must be in [0, 1]'
        assert patch_size & (patch_size - 1) == 0, '"patch_size" must be power of 2'
        assert isinstance(n_append, int) and n_append >= 1, '"n_append" must be a positive int'
        was_2d = (len(img.shape) == 2)
        if len(img.shape) == 3:
            assert img.shape[0] in [3, 4], \
                "colored image must be in shape (C, H, W) with C in [3, 4]"
        else:
            assert len(img.shape) == 2, "grey image must be in shape (H, W)"
            img = np.expand_dims(img, 0)

        self.patch_size = patch_size
        k_in = int(math.log2(patch_size))
        n_qubit = (k_in + n_append) * 2      # qubits of the enlarged (output) register
        assert 4 <= n_qubit <= 20, \
            f'invalid output "n_qubit": {n_qubit}; reduce patch_size or n_append'
        self.n_qubit = n_qubit
        self._scale = 2 ** n_append

        C, H, W = img.shape
        # pad to whole patches, [C, H, W]
        img_pad, pad_rs = self.pad(img)
        # split channels, C * [H, W]
        channels = [img_pad[c] for c in range(C)]
        channels_processed: list[ndarray] = []
        for channel in channels:
            # patchify, [B, p, p]
            patches = self.pachify(channel)
            # vectorize, [B, d=p**2]
            vectors, norm = self.vectorize(patches)
            # QFT-append-IQFT (spectral zero-pad), [B, D=P**2]
            qvectors = self.apply_unQJPEG(vectors, n_append)
            # devectorize, [B, P, P]
            devectors = self.devectorize(qvectors, norm)
            # unpatchify, [Ho, Wo]
            channel_unpatchify = self.unpachify(devectors)
            channels_processed.append(channel_unpatchify)
        # merge channel, [C, Ho, Wo]
        img_merged = np.stack(channels_processed, axis=0)
        # trim pads, [C, Ho, Wo]
        img_unpad = self.unpad(img_merged, pad_rs)
        return img_unpad[0] if was_2d else img_unpad

    def pad(self, img: ndarray):
        """Zero-pad an ``[C, H, W]`` image so H and W are whole multiples of
        ``patch_size``, pasting the content to the center. Returns the padded image
        and the ``(top, bottom, left, right)`` pad widths, or ``None`` if unneeded."""
        C, H, W = img.shape
        n_patch_H = math.ceil(H / self.patch_size)
        n_patch_W = math.ceil(W / self.patch_size)
        H_ex = n_patch_H * self.patch_size
        W_ex = n_patch_W * self.patch_size
        if H_ex == H and W_ex == W:
            return img, None
        pad_H = H_ex - H
        pad_W = W_ex - W
        pads = (
            math.floor(pad_H / 2), math.ceil(pad_H / 2),
            math.floor(pad_W / 2), math.ceil(pad_W / 2),
        )
        img_ex = np.zeros(shape=(C, H_ex, W_ex), dtype=img.dtype)
        img_ex[:, pads[0]:pads[0] + H, pads[2]:pads[2] + W] = img
        return img_ex, pads

    def unpad(self, img: ndarray, pads=None) -> ndarray:
        """Trim the padding added by :meth:`pad`. The pad widths are scaled *up* by
        the up-sampling factor to match the enlarged resolution."""
        if not pads or all(e == 0 for e in pads):
            return img
        top, bottom, left, right = pads
        s = self._scale
        C, H, W = img.shape
        t = round(top * s)
        b = round(bottom * s)
        l = round(left * s)
        r = round(right * s)
        return img[:, t:(H - b) if b else H, l:(W - r) if r else W]

    def pachify(self, img: ndarray) -> ndarray:
        """Split a single-channel ``[H, W]`` image into non-overlapping
        ``patch_size`` x ``patch_size`` patches, returned as ``[B, p, p]``."""
        P = self.patch_size
        H, W = img.shape
        assert H % P == 0 and W % P == 0
        nH, nW = H // P, W // P
        self._grid = (nH, nW)
        return img.reshape(nH, P, nW, P).swapaxes(1, 2).reshape(nH * nW, P, P)

    def unpachify(self, patches: ndarray) -> ndarray:
        """Re-assemble ``[B, P, P]`` (up-sampled) patches into a ``[Ho, Wo]`` image."""
        nH, nW = self._grid
        B, P, _ = patches.shape
        assert B == nH * nW
        return patches.reshape(nH, nW, P, P).swapaxes(1, 2).reshape(nH * P, nW * P)

    def vectorize(self, patches: ndarray) -> tuple:
        """Vectorize ``[B, p, p]`` patches to normalized amplitude vectors
        ``[B, p**2]`` (``amplitude = sqrt(pixel / patch_sum)``) and return the
        per-patch intensity ``norm``."""
        B = patches.shape[0]
        vect = patches.reshape(B, -1).astype(np.float64)
        norm = vect.sum(axis=1)
        amps = np.zeros_like(vect)
        nz = norm > 0
        amps[nz] = np.sqrt(vect[nz] / norm[nz, None])
        return amps, norm

    def devectorize(self, vectors: ndarray, norm: object) -> ndarray:
        """Decode output probability vectors ``[B, P**2]`` into intensity patches
        ``[B, P, P]``, rescaled by ``norm`` and the pixel-count ratio so the mean
        intensity of each patch is preserved by the up-sampling."""
        vectors = np.asarray(vectors)
        B, D = vectors.shape
        P = int(round(math.sqrt(D)))
        n_in = self.patch_size ** 2
        n_out = D
        norm = np.asarray(norm)
        patches = vectors * norm[:, None] * (n_out / n_in)
        return patches.reshape(B, P, P)

    def apply_unQJPEG(self, vectors: ndarray, n_append: int) -> ndarray:
        """Run the unQJPEG (spectral zero-pad) circuit on each amplitude vector
        ``[B, p**2]`` and return the up-sampled probability vectors ``[B, P**2]``,
        with ``P = p * 2 ** n_append``."""
        k2 = self.n_qubit // 2 - n_append   # qubits per axis of the (small) input
        k0 = k2 + n_append                  # qubits per axis of the (large) output
        n0 = 2 * k0
        D = 2 ** n0

        # per-axis qubit layout of the enlarged register
        col_low = list(range(0, k2 - 1))
        col_mid = list(range(k2 - 1, k0 - 1))
        col_msb = k0 - 1
        row_low = list(range(k0, k0 + k2 - 1))
        row_mid = list(range(k0 + k2 - 1, 2 * k0 - 1))
        row_msb = 2 * k0 - 1
        # data index bit order: col bits (LSB..MSB) then row bits (LSB..MSB)
        enc_qubits = col_low + [col_msb] + row_low + [row_msb]
        col_small = col_low + [col_msb]
        row_small = row_low + [row_msb]

        vectors = np.asarray(vectors)
        B = vectors.shape[0]
        out = np.zeros((B, D), dtype=np.float64)
        rng = np.random.default_rng(self.seed) if self.shots else None

        for b in range(B):
            vec = vectors[b]
            if not np.any(vec):     # empty patch -> stays empty
                continue
            enc = Encode()
            enc.amplitude_encode(enc_qubits, list(vec))
            prog = QProg(n0)
            prog << enc.get_circuit()
            if self.wrap_H_layer:
                for qi in (col_small + row_small):
                    prog << H(qi)
            # per-axis QFT of the small registers
            prog << QFT(col_small)
            prog << QFT(row_small)
            # zero-pad the spectrum: copy the sign qubit onto the appended qubits
            for qi in col_mid:
                prog << CNOT(col_msb, qi)
            for qi in row_mid:
                prog << CNOT(row_msb, qi)
            # per-axis inverse QFT of the enlarged registers
            prog << QFT(list(range(0, k0))).dagger()
            prog << QFT(list(range(k0, 2 * k0))).dagger()
            if self.wrap_H_layer:
                for qi in range(n0):
                    prog << H(qi)
            machine = CPUQVM()
            machine.run(prog, 1)
            prob_dict = machine.result().get_prob_dict(list(range(n0)))
            probs = _prob_dict_to_vector(prob_dict, n0, D)
            if self.shots:
                probs = rng.multinomial(self.shots, probs) / self.shots
            out[b] = probs
        return out


def _prob_dict_to_vector(prob_dict: dict, n: int, dim: int) -> ndarray:
    """Convert a ``get_prob_dict`` result over qubits ``0..n-1`` into a dense
    vector indexed with qubit ``0`` as the least-significant bit."""
    vec = np.zeros(dim, dtype=np.float64)
    for key, val in prob_dict.items():
        idx = 0
        for k in range(n):
            if key[n - 1 - k] == '1':
                idx |= (1 << k)
        vec[idx] += val
    return vec
