import tqdm
import math
import numpy as np

from scipy.sparse import csr_matrix


def sdmm_reordering(matrixA_csr: csr_matrix, matrixB_dense: np.ndarray=None, tile_size: int=512, leave_pbar: bool=False, only_dimM: bool=False):
    a = matrixA_csr.data
    ptr_a = matrixA_csr.indptr.astype(np.dtype(np.int32))
    idx_a = matrixA_csr.indices.astype(np.dtype(np.int32))
    
    num_nodes = len(ptr_a) - 1
    
    reordered_rowA_idx = []
    remaining_rowA_idx = list(range(num_nodes))
    
    for tile_idx in tqdm.tqdm(range(math.ceil(num_nodes / tile_size)), ncols=100, desc="compute index   ", leave=leave_pbar):
        selected = []

        for i in remaining_rowA_idx:
            st_ptr  = ptr_a[i]
            ed_ptr  = ptr_a[i+1]
            idx_arr = idx_a[st_ptr:ed_ptr]
            
            selected.append((i, np.count_nonzero(np.logical_and((tile_idx * tile_size) <= idx_arr, idx_arr < ((tile_idx + 1) * tile_size)))))
            
        selected = sorted(selected, key=lambda x: x[1], reverse=True)
        
        for r in selected[:tile_size]:
            reordered_rowA_idx.append(r[0])
            remaining_rowA_idx.remove(r[0])
            
    reordered_rowA_idx = np.array(reordered_rowA_idx)

    matrixA_dense = matrixA_csr.todense()

    reord_matrixA_dense = np.zeros_like(matrixA_dense)
    for orig_i, reord_i in enumerate(tqdm.tqdm(reordered_rowA_idx, ncols=100, desc="reorder matrixA ", leave=leave_pbar)):
        if only_dimM:
            reord_matrixA_dense[orig_i, :] = matrixA_dense[reord_i, :]
        else:
            reord_matrixA_dense[orig_i, :] = matrixA_dense[reord_i, reordered_rowA_idx]
        
    reord_matrixA_csr = csr_matrix(reord_matrixA_dense)
    
    if matrixB_dense is not None:
        if only_dimM:
            reord_matrixB_dense = matrixB_dense
        else:  
            reord_matrixB_dense = np.zeros_like(matrixB_dense)
            
            for orig_i, reord_i in enumerate(tqdm.tqdm(reordered_rowA_idx, ncols=100, desc="reorder matrixB ", leave=leave_pbar)):
                reord_matrixB_dense[orig_i, :] = matrixB_dense[reord_i, :]
    
    if matrixB_dense is not None:
        return reord_matrixA_csr, reord_matrixB_dense
    else:
        return reord_matrixA_csr
    