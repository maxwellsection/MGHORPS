import numpy as np
import scipy.sparse as sp
import time
import threading
from typing import List, Tuple

class NanoSlice:
    """表示一个送入 NPU SRAM 的矩阵微小切片 (Nano-slice)"""
    def __init__(self, slice_id: int, row_start: int, row_end: int, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray):
        self.slice_id = slice_id
        self.row_start = row_start
        self.row_end = row_end
        self.data = data
        self.indices = indices
        self.indptr = indptr

class NPUEdgeScheduler:
    """
    针对边缘设备 (Rockchip/Horizon 等 NPU) 的纳米切片调度器。
    
    这类设备的特征是：
    1. MAC (乘加) 算力极强
    2. SRAM (片上缓存) 极小 (例如只有 1MB - 4MB)
    3. 不擅长逻辑分支和大型内存随机访问
    
    调度策略：将大型高度稀疏的约束矩阵 $A$ 切片成 NPU SRAM 可容纳的大小，流水线推入执行块。
    由于我们在 PC 上仿真，使用线程池模拟异步推入硬件流水线。
    """
    
    def __init__(self, sram_size_kb: int = 512, num_npu_cores: int = 2):
        self.sram_size_bytes = sram_size_kb * 1024
        self.num_npu_cores = num_npu_cores
        self._cached_slices = {}
        print(f"📱 初始化边缘 NPU 调度仿真器")
        print(f"   - 虚拟 SRAM 大小: {sram_size_kb} KB")
        print(f"   - 虚拟 NPU 核心数: {num_npu_cores}")
        
    def _slice_matrix(self, A_csr: sp.csr_matrix) -> List[NanoSlice]:
        """将 CSR 稀疏矩阵按照行切分为适合 SRAM 的纳米切片"""
        slices = []
        n_rows = A_csr.shape[0]
        
        # 简单粗暴的切片策略：根据数据量大小划分
        # 每行 CSR 占用的字节大约是 nnz * (8 byte (data) + 4 byte (indices)) + 4 byte (indptr)
        bytes_per_nnz = 12
        bytes_per_row_base = 4
        
        current_rows = 0
        current_size = 0
        start_row = 0
        slice_id = 0
        
        for i in range(n_rows):
            nnz_in_row = A_csr.indptr[i+1] - A_csr.indptr[i]
            row_size = nnz_in_row * bytes_per_nnz + bytes_per_row_base
            
            if current_size + row_size > self.sram_size_bytes and current_rows > 0:
                # 生成一个切片
                A_slice = A_csr[start_row:i, :]
                slices.append(NanoSlice(
                    slice_id=slice_id,
                    row_start=start_row,
                    row_end=i,
                    data=A_slice.data,
                    indices=A_slice.indices,
                    indptr=A_slice.indptr
                ))
                slice_id += 1
                start_row = i
                current_size = 0
                current_rows = 0
            
            current_size += row_size
            current_rows += 1
            
        # 最后一个切片
        if current_rows > 0:
            A_slice = A_csr[start_row:n_rows, :]
            slices.append(NanoSlice(
                slice_id=slice_id,
                row_start=start_row,
                row_end=n_rows,
                data=A_slice.data,
                indices=A_slice.indices,
                indptr=A_slice.indptr
            ))
            
        print(f"   ✂️ 矩阵切分成 {len(slices)} 个 Nano-slices")
        return slices

    def _simulate_npu_mac(self, nano_slice: NanoSlice, vector: np.ndarray, result_out: np.ndarray):
        """模拟 NPU 核心上的纯 MAC 运算，无逻辑分支"""
        # 我们使用 numpy 内部的高效运算来近似 C 语言底层的循环展开 MAC 运算
        n_slice_rows = nano_slice.row_end - nano_slice.row_start
        # 重建临时 CSR (在真实硬件中，这里是针对特定格式的 NPU 汇编)
        temp_csr = sp.csr_matrix(
            (nano_slice.data, nano_slice.indices, nano_slice.indptr), 
            shape=(n_slice_rows, vector.shape[0])
        )
        
        # MAC (Multiply-Accumulate)
        res = temp_csr.dot(vector)
        
        # 写回内存
        result_out[nano_slice.row_start:nano_slice.row_end] += res

    def async_spmv(self, A_csr: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        """
        异步稀疏矩阵向量乘 (Asynchronous Sparse Matrix-Vector Multiplication)
        此方法用于在 PDHG 中替换标准的 A * x 运算。
        """
        matrix_id = id(A_csr)
        if matrix_id not in self._cached_slices:
            self._cached_slices[matrix_id] = self._slice_matrix(A_csr)
            
        slices = self._cached_slices[matrix_id]
        result = np.zeros(A_csr.shape[0])
        
        # 真实环境这里会是将任务推入设备队列，我们在仿真使用线程池
        threads = []
        for s in slices:
            # 模拟推入 NPU Core 执行
            t = threading.Thread(target=self._simulate_npu_mac, args=(s, x, result))
            threads.append(t)
            t.start()
            
            # 限制并发核心数模拟 NPU 并行度
            if len(threads) >= self.num_npu_cores:
                for active_t in threads:
                    active_t.join()
                threads = []
                
        # 等待剩余任务完成
        for active_t in threads:
            active_t.join()
            
        return result
