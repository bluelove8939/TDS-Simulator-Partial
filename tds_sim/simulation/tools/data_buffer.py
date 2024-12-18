import torch
import numpy as np

from tds_sim.common.data_tools import cast2bytearr, cast2anytype
from tds_sim.common.custom_exception import CustomException


def _bin_search(arr, target) -> int:
    st, ed, cen = 0, len(arr), len(arr) // 2
    
    while cen - st > 0:
        if arr[cen] == target:
            break
        elif arr[cen] < target:
            st = cen
        else:
            ed = cen
        cen = (st + ed) // 2
        
    return cen  # returns index of the searched item


class SingleBankBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        
        self._memory = np.zeros(size=self.capacity, dtype=torch.uint8)
        
    def get_capacity(self):
        return self._capacity
        
    def get_data(self, addr: int, size: int, dtype: np.dtype) -> np.ndarray:
        return cast2anytype(self._memory[addr:addr+size], dtype=dtype)
    
    def set_data(self, addr: int, data: np.ndarray):
        size = data.size * data.itemsize
        self._memory[addr:addr+size] = cast2bytearr(data)


class MultiBankBuffer(object):
    def __init__(self, capacity: int, bank_num: int) -> None:
        if capacity % bank_num:
            raise CustomException(self, f"the capacity({capacity}) should be the multiple of the bank number({bank_num})")
        
        self._capacity = capacity
        self._bank_num = bank_num
        self._bank_siz = capacity // bank_num
        
        self._memory = np.zeros(size=(self._bank_num, self._bank_siz), dtype=torch.uint8)
        
    def get_capacity(self):
        return self._capacity
    
    def get_bank_num(self):
        return self._bank_num
    
    def get_bank_siz(self):
        return self._bank_siz
        
    def get_data(self, bank_id: int, bank_offset: int, size: int, dtype: np.dtype) -> np.ndarray:
        if (bank_offset + size > self._bank_siz):
            raise CustomException(self, f"bank address({bank_offset}-{bank_offset+size}) exceeds the bank size({self._bank_siz})")
        return cast2anytype(self._memory[bank_id, bank_offset:bank_offset+size], dtype=dtype)
    
    def set_data(self, bank_id: int, bank_offset: int, data: np.ndarray):
        size = data.size * data.itemsize
        self._memory[bank_id, bank_offset:bank_offset+size] = cast2bytearr(data)
        
        
class OffchipMainMemory(object):
    def __init__(self) -> None:
        self._data_segments: dict[int, np.ndarray[np.uint8]] = {}
        
    def get_data(self, addr: int, size: int, dtype: np.dtype) -> np.ndarray:
        offsets = list(sorted(self._data_segments.keys()))
        # data = np.array([], dtype=np.uint8)
        data_fragments = []
        
        while True:
            idx = _bin_search(offsets, addr)
            
            if idx >= len(offsets):
                return cast2anytype(np.zeros(size, dtype=np.uint8), dtype=dtype)
            
            offset = offsets[idx]
            dist = addr - offset
            segment = self._data_segments[offset]
            segment_size_left = len(segment) - dist
            
            if segment_size_left <= 0:
                data_fragments.append(np.zeros(size, dtype=np.uint8))
                break
            if size <= segment_size_left:
                data_fragments.append(self._data_segments[offset][dist:dist+size])
                break
            else:
                new_addr = addr + segment_size_left
                new_size = size - segment_size_left
                
                if (addr, size) == (new_addr, new_size):
                    raise CustomException(self, "same recursion call found! (maybe there must be faulty implementation...)")
                
                data_fragments.append(self._data_segments[offset][dist:dist+segment_size_left])
                
                addr = new_addr
                size = new_size
        
        if len(data_fragments):
            data = np.concatenate(data_fragments)
        else:
            data = np.zeros(size, dtype=np.uint8)
        
        return cast2anytype(data, dtype)
    
    def set_data(self, addr: int, data: np.ndarray):
        data = cast2bytearr(data.flatten())
        offsets = list(sorted(self._data_segments.keys()))
        size = data.size
        
        while True:
            idx = _bin_search(offsets, addr)
            
            if idx >= len(offsets):
                self._data_segments[addr] = data
                break
            
            offset = offsets[idx]
            dist = addr - offset
            segment = self._data_segments[offset]
            segment_size_left = len(segment) - dist
            
            if segment_size_left <= 0:
                self._data_segments[addr] = data
                break
            if size <= segment_size_left:
                self._data_segments[offset][dist:dist+size] = data[:size]
                data = data[size:]
                break
            else:
                new_addr = addr + segment_size_left
                new_size = size - segment_size_left
                
                if (addr, size) == (new_addr, new_size):
                    raise CustomException(self, "same recursion call found! (maybe there must be faulty implementation...)")
                
                self._data_segments[offset][dist:dist+segment_size_left] = data[:segment_size_left]
                
                addr = new_addr
                size = new_size
                
        self.compact_buffer_segments()  # TODO: check compatibility
                
    def compact_buffer_segments(self):
        offsets = list(sorted(self._data_segments.keys()))
        pivot = offsets[0]
        end_point = pivot + self._data_segments[pivot].size
        
        for offset in offsets[1:]:
            if end_point == offset:
                segment = self._data_segments[offset]
                end_point += segment.size
                self._data_segments[pivot] = np.concatenate((self._data_segments[pivot], segment))
                
                self._data_segments.pop(offset)
            else:
                pivot = offset
                end_point = pivot + self._data_segments[pivot].size

        
if __name__ == "__main__":
    ofc_mem = OffchipMainMemory()
        
    ofc_mem.set_data(0,  np.array([1, 2, 3, 4], dtype=np.int32))
    ofc_mem.set_data(0,  np.array([1, 2, 3, 4], dtype=np.uint8))
    ofc_mem.set_data(16, np.array([1, 2, 3, 4], dtype=np.int32))
    ofc_mem.set_data(32, np.array([1, 2, 3, 4], dtype=np.int32))
    ofc_mem.set_data(48, np.array([1, 2, 3, 4], dtype=np.int32))
    
    print(ofc_mem.get_data(addr=4, size=56, dtype=np.int32))
    print(ofc_mem.get_data(addr=56, size=32, dtype=np.int32))
    
    for offset in list(sorted(ofc_mem._data_segments.keys())):
        data = ofc_mem._data_segments[offset]
        print(f"   * {offset:3d} - {offset+data.size:3d} >> {data}")  