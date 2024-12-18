import struct
import torch
import ctypes
import math
import numpy as np

from typing import Any


def cast2bytearr(arr):
    if isinstance(arr, np.ndarray):
        buffer = arr.tobytes()
        return np.copy(np.frombuffer(buffer=buffer, dtype=np.uint8))
    # elif isinstance(arr, torch.Tensor):
    #     return arr.clone().view(torch.uint8)
    else:
        raise Exception(f"[ERROR] cast2anytype requires array with its type of numpy.ndarray or torch.Tensor, not '{type(arr).__name__}'")
    
def cast2anytype(arr, dtype):
    if isinstance(arr, np.ndarray):
        # if not isinstance(dtype, np.dtype):
        #     raise Exception(f"[ERROR] cast2anytype for numpy requires dtype of 'numpy.dtype', not '{type(dtype).__name__}'")
        
        buffer = arr.tobytes()
        return np.copy(np.frombuffer(buffer=buffer, dtype=dtype))
    # elif isinstance(arr, torch.Tensor):
    #     return arr.clone().view(dtype)
    else:
        raise Exception(f"[ERROR] cast2anytype requires array with its type of numpy.ndarray or torch.Tensor, not '{type(arr).__name__}'")

def float_cast2int(val: float):
    b = struct.pack('<f', val)
    return struct.unpack('<i', b)

def int_cast2float(val: int):
    b = struct.pack('<i', val)
    return struct.unpack('<f', b)

def ifm_lowering(tensor: torch.Tensor, weight_shape: tuple[int], padding: int, stride: int):
    N, C, H, W = tensor.shape
    _, _, FW, FH = weight_shape
    
    OH = (H - FH + (2 * padding)) // stride + 1
    OW = (W - FW + (2 * padding)) // stride + 1

    if padding > 0:
        tensor = torch.nn.functional.pad(tensor, (padding, padding, padding, padding, 0, 0, 0, 0), 'constant', value=0)
        
    tensor = tensor.permute((0, 2, 3, 1))  # N, H, W, C
    output_tensor = torch.zeros(size=(N, OH, OW, FH * FW * C), dtype=tensor.dtype)

    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                h = oh * stride
                w = ow * stride
                output_tensor[n, oh, ow, :] = tensor[n, h:h+FH, w:w+FW, :].flatten()

    return output_tensor.reshape((N*OH*OW, FH*FW*C))

def wgt_lowering(tensor: torch.Tensor):
    K, _, _, _ = tensor.shape
    tensor = tensor.permute((0, 2, 3, 1))  # K, FH, FW, C
    return tensor.reshape((K, -1)).T

def get_size_from_anytype(self, value: Any):
    if value is None:
        return 1
    elif isinstance(value, int):
        return math.ceil(math.log2(abs(value)))
    elif isinstance(value, float):
        return 32
    elif isinstance(value, np.ndarray):
        return value.size * value.itemsize
    elif isinstance(value, torch.Tensor):
        return value.size() * value.dtype.itemsize
    else:
        return 0
