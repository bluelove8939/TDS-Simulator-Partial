import numpy as np


def quantize(arr: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, float]:
    emax = np.max(np.abs(arr))
    nbin = 2 ** (dtype.itemsize * 8 - 1)
    
    scale = emax / nbin
    
    quantized_arr = np.round(arr / scale, 0).astype(dtype)
    
    return quantized_arr, scale


def dequantize(quantized_arr: np.ndarray, scale: float, dtype: np.dtype) -> np.ndarray:
    return quantized_arr.astype(dtype) * scale