import numpy as np
from numpy import ndarray


class unQJPEG:

  def __init__(self, wrap_H_layer:bool=True):
    pass

  def __call__(self, img:ndarray, n_append:int=1, patch_size:int=128):
    # check image valid grey or RGB
    # check n_discard positive and not too large
    # check patch_size not too large
    #   patch_size 32x32 -> 10 qubits
    #   patch_size 128x128 -> 14 qubits
    #   patch_size 512x512 -> 18 qubits
    n_qubit = int(np.log2(patch_size)) * 2
    
    # pad to square      [C, H, W]
    # split channel      C * [H, W]
    # patchify           C * [B=(H*W/P**2), h=P, w=P]
    # vectorize          C * B=(H*W/P**2) * [D=P**2]
    # QFT-discard-IQFT   C * B=(H*W/P**2) * [d=p**2]
    # devectorize
    # unpatchify
    # merge channel
    # unpad to original size
    raise NotImplementedError
