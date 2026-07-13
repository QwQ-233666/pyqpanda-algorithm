import math
import numpy as np
from numpy import ndarray


class QJPEG:

    def __init__(self, patch_size:int=256, wrap_H_layer:bool=True):
        assert patch_size & (patch_size - 1) == 0, '"patch_size" must be power of 2'
        n_qubit = int(math.log2(patch_size)) * 2
        assert 4 <= n_qubit <= 20, f'invalid "n_qubit": {n_qubit}'

        self.patch_size = patch_size
        self.n_qubit = n_qubit
        self.wrap_H_layer = wrap_H_layer

    def __call__(self, img:ndarray, n_discard:int=2) -> ndarray:
        assert isinstance(img, ndarray)
        assert img.dtype in [np.float32, np.float64], f'invalid "n_discard": {n_discard}'
        assert 0 <= img.min() and img.max() <= 1.0
        if len(img.shape) == 3:
            assert img.shape[0] in [3, 4], "colored image must be in shape (C, H, W) with C in [3, 4]"
        else:
            assert len(img.shape) == 2, "grey image must be in shape (H, W)"
            img = np.expand_dims(img, 0)
        C, H, W = img.shape
        assert isinstance(n_discard, int) and n_discard > 0, f'invalid "n_discard": {n_discard}'

        # pad to square, [C, H, W]
        img_pad, pad_rs = self.pad(img)
        # split channels, C * [H, W]
        channels = [img_pad[c] for c in range(C)]
        channels_processed: list[ndarray] = []
        # foreach C
        for channel in channels:
            # patchify, [B=(H*W/P**2), h=P, w=P]
            patches = self.pachify(channel)
            # vectorize, [B=(H*W/P**2), D=P**2]
            vectors, norm = self.vectorize(patches)
            # QFT-discard-IQFT, [B=(H*W/P**2), d=p**2]
            qvectors = self.apply_QJPEG(vectors)
            # devectorize, [B=(H*W/P**2), d']
            devectors = self.devectorize(qvectors, norm)
            # unpatchify, [Ho, Wo]
            channel_unpatchify = self.unpachify(devectors)
            # collect
            channels_processed.append(channel_unpatchify)
        # merge channel, [C, Ho, Wo]
        img_merged = np.stack(channels_processed, axis=0)
        # trim pads, [C, Ho, Wo]
        img_unpad = self.unpad(img_merged, pad_rs)
        return img_unpad

    def pad(self, img:ndarray) -> tuple[ndarray, tuple[float, float, float, float]]:
        C, H, W = img.shape
        n_patch_H = math.ceil(H / self.patch_size)
        n_patch_W = math.ceil(W / self.patch_size)
        H_ex = n_patch_H * self.patch_size
        W_ex = n_patch_W * self.patch_size
        if H_ex == H and W_ex == W: return img, None
        pad_H = H_ex - H
        pad_W = W_ex - W
        pad_rs = [
            math.floor(pad_H / 2) / H,
            math.ceil (pad_H / 2) / H,
            math.floor(pad_W / 2) / W,
            math.ceil (pad_W / 2) / W,
        ]
        # paste to center
        img_ex = np.zeros(shape=(C, H_ex, W_ex), dtype=img.dtype)
        img_ex[:, ] = img
        return img_ex, pad_rs

    def unpad(self, img:ndarray, pad_rs:tuple[float, float, float, float]=None) -> ndarray:
        if not pad_rs or all(e == 0 for e in pad_rs): return img
        C, H, W = img.shpe
        a, b, c, d = pad_rs
        to_lb = lambda x: math.ceil(x)
        to_ub = lambda x: None if x <= 0 else -math.ceil(x)
        return img[to_lb(a * H) : to_ub(b * H), to_lb(c * W) : to_ub(d * W)]

    def pachify(self, img:ndarray) -> ndarray:
        pass

    def unpachify(self, pacthes:ndarray) -> ndarray:
        pass

    def vectorize(self, patches:ndarray) -> tuple[ndarray, object]:
        pass

    def devectorize(self, vectors:ndarray, norm:object) -> ndarray:
        pass

    def apply_QJPEG(self, vectors:ndarray) -> ndarray:
        pass
