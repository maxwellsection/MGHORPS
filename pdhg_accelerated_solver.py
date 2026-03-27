import numpy as np
import scipy.sparse as sp
import time
from typing import Dict, Any, Tuple

from npu_edge_scheduler import NPUEdgeScheduler

class PDHGSolver:
    """
    Primal-Dual Hybrid Gradient (Chambolle-Pock) 求解器引擎
    基于梯度的极速解法，时间复杂度与非零元素数量 O(nnz(A)) 成正比。
    极其适合 GPU/NPU 等并行硬件。
    
    求解标准形式：
    min c^T x
    s.t. Ax = b  (或 Ax <= b)
         l <= x <= u
    """
    def __init__(self, tolerance: float = 1e-5, max_iter: int = 100000, use_npu: bool = False, npu_cores: int = 2, verbose_options: dict = None):
        self.verbose_options = verbose_options or {'basic': True, 'iterations': False}
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.use_npu = use_npu
        if self.use_npu:
            self.npu_scheduler = NPUEdgeScheduler(num_npu_cores=npu_cores)
        
    def _compute_step_sizes(self, A: sp.csc_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算主问题和对偶问题的步长 (Primal and Dual step sizes)
        为了保证 PDHG 收敛，需要 tau * sigma * ||A||^2 < 1
        使用行范数和列范数进行对角预处理平衡
        """
        # 计算每一列的 L1 范数，用于主变量步长 tau
        col_norms = np.array(np.abs(A).sum(axis=0)).flatten()
        col_norms[col_norms == 0] = 1.0  # 防止除以0
        tau = 1.0 / col_norms
        
        # 计算每一行的 L1 范数，用于对偶变量步长 sigma
        row_norms = np.array(np.abs(A).sum(axis=1)).flatten()
        row_norms[row_norms == 0] = 1.0
        sigma = 1.0 / row_norms
        
        return tau, sigma

    def _log(self, logs: dict, category: str, msg: str):
        if category not in logs: logs[category] = []
        logs[category].append(msg)
        if self.verbose_options.get(category, False):
            print(msg)
            
    def solve(self, c: np.ndarray, A: sp.csc_matrix, b: np.ndarray, 
              bounds: np.ndarray, constraint_types: np.ndarray) -> Dict[str, Any]:
        logs = {'basic': [], 'iterations': []}
        """
        运行 PDHG 迭代
        
        参数:
            c: 目标函数系数 (长度 n)
            A: 稀疏约束矩阵 (m x n)
            b: 约束右端项 (长度 m)
            bounds: 变量边界 (n x 2) -> [[低界, 高界], ...]
            constraint_types: 长度m，标记该约束是否为等式('=':1, '<=':0)
        """
        self._log(logs, 'basic', f"🚀 开始 PDHG (Primal-Dual Hybrid Gradient) 求解迭代...")
        self._log(logs, 'basic', f"   📐 矩阵规模: {A.shape[0]}行 x {A.shape[1]}列, 非零元素: {A.nnz}")
        
        m, n = A.shape
        start_time = time.time()
        
        # 将矩阵转为基于列和基于行的格式以加快稀疏乘法计算速度
        A_csc = A.tocsc()
        A_csr = A.tocsr()
        A_T_csr = A.transpose().tocsr()
        
        # 1. 计算步长
        tau, sigma = self._compute_step_sizes(A_csc)
        
        # 2. 初始化变量
        # 主变量 (Primal variables)
        x = np.zeros(n)
        x_bar = np.zeros(n) # 外推预测
        
        # 对偶变量 (Dual variables / 拉格朗日乘子)
        y = np.zeros(m)
        
        # 近似矩阵范数界限
        theta = 1.0 
        
        iteration = 0
        status = 'max_iterations_reached'
        
        while iteration < self.max_iter:
            x_old = x.copy()
            y_old = y.copy()
            
            # --- Primal Update (主变量更新) ---
            # x_next = Proj_X (x - tau * (c + A^T y))
            # 注意：A^T y 是由于求梯度的计算
            if self.use_npu:
                grad_x = c + self.npu_scheduler.async_spmv(A_T_csr, y)
            else:
                grad_x = c + A_T_csr.dot(y)
            x = x - tau * grad_x
            
            # 投影到主变量边界 [l, u] 上
            # Bounds 是Nx2 数组，如果是 None 或者无界则为 np.inf
            lower_bounds = bounds[:, 0]
            upper_bounds = bounds[:, 1]
            x = np.clip(x, lower_bounds, upper_bounds)
            
            # --- Extrapolation (外推，用于加速收敛和保证稳定性) ---
            x_bar = x + theta * (x - x_old)
            
            # --- Dual Update (对偶变量更新) ---
            # y_next = Proj_Y (y + sigma * (A x_bar - b))
            if self.use_npu:
                grad_y = self.npu_scheduler.async_spmv(A_csr, x_bar) - b
            else:
                grad_y = A_csr.dot(x_bar) - b
            y = y + sigma * grad_y
            
            # 投影到对偶边界上
            # 对于等式约束 (=)，对偶变量没有约束限制
            # 对于不等式约束 (<=)，由于引入拉格朗日乘子，如果 b - Ax >= 0，则 y 的方向需要 >=0 (通常定义为非负)
            # 根据原问题是最小化，拉格朗日乘子 y >= 0 或 y <= 0 取决于符号定义
            # 标准PDLP通常将投影逻辑分给 slack 变量，我们直接处理 y: y >= 0 for '<=' constraints
            for i in range(m):
                if constraint_types[i] == 0:  # <=
                    y[i] = max(0.0, y[i])
            
            # --- 收敛判定 ---
            if iteration > 0 and iteration % 1000 == 0:
                # 检查主残差 KKT 条件 (Ax - b 的侵犯程度)
                ax = A_csr.dot(x)
                primal_residual = 0.0
                for i in range(m):
                    if constraint_types[i] == 1: # ==
                        primal_residual += abs(ax[i] - b[i])
                    else: # <=
                        primal_residual += max(0.0, ax[i] - b[i])
                
                # 目标值差异
                obj_val = c.dot(x)
                
                dx_norm = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x))
                dy_norm = np.linalg.norm(y - y_old) / max(1.0, np.linalg.norm(y))
                
                if primal_residual < self.tolerance and dx_norm < self.tolerance and dy_norm < self.tolerance:
                    self._log(logs, 'iterations', f"      ✅ PDHG 在迭代 {iteration} 达到容差要求。 (dx_norm: {dx_norm:.2e}, dy_norm: {dy_norm:.2e})")
                    status = 'optimal'
                    break
                    
                if iteration % 10000 == 0:
                    self._log(logs, 'iterations', f"      📊 PDHG 迭代 {iteration:5d} - 目标值: {obj_val:.4f}, 主约束残差: {primal_residual:.2e}, dx_norm: {dx_norm:.2e}")
            
            iteration += 1

        solve_time = time.time() - start_time
        obj_value = c.dot(x)
        
        return {
            'status': status,
            'solution': x,
            'objective_value': obj_value,
            'solve_time': solve_time,
            'iterations': iteration,
            'logs': logs
        }
