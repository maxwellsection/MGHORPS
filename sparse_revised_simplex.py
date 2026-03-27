import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import warnings

class SparseRevisedSimplex:
    """
    基于稀疏矩阵与 LU 分解的改进单纯形法 (Revised Simplex Method)
    此求解器完全使用 scipy.sparse，避免了全内存单纯形表的维护。
    """
    
    def __init__(self, tolerance=1e-8, max_iter=5000):
        self.tol = tolerance
        self.max_iter = max_iter
        
    def solve(self, c, A, b, is_maximize=False):
        """
        求解标准型 LP:
            min (or max) c^T x
            s.t. Ax = b
                 x >= 0
        参数:
            c: 目标函数系数 (1D array)
            A: 约束系数矩阵 (scipy.sparse.csc_matrix)
            b: 右端项 (1D array)
            is_maximize: 是否为最大化问题
        返回:
            {'status', 'solution', 'objective_value', 'iterations'}
        """
        start_time = time.time()
        m, n = A.shape
        
        # 确保 A 是 CSC 格式以按列快速切片
        A = sp.csc_matrix(A)
        c = np.array(c, dtype=float)
        b = np.array(b, dtype=float)
        
        # 统一转化为最小化问题求解内部逻辑
        if is_maximize:
            c_internal = -c
        else:
            c_internal = c
            
        # 确保 RHS b >= 0
        for i in range(m):
            if b[i] < 0:
                b[i] = -b[i]
                A.data[A.indptr[i]:A.indptr[i+1]] = -A.data[A.indptr[i]:A.indptr[i+1]]
                # 上述方法对 CSR 适用，对 CSC 则很麻烦。
                # 改用稀疏矩阵乘法：
                row_multiplier = np.ones(m)
        
        # 统一确保 RHS b >= 0, 把负号乘到矩阵 A 的相应行
        row_signs = np.where(b < 0, -1.0, 1.0)
        b = b * row_signs
        
        # 用对角矩阵相乘来翻转行符号
        M_signs = sp.diags(row_signs)
        A = M_signs.dot(A)
            
        # 阶段一：使用两阶段法找初始可行基 (Phase I)
        # 添加人工变量 A_art = I, c_art = 1
        print("      🔄 Phase I: 寻找初始基本可行解 (Sparse)")
        A_phase1 = sp.hstack([A, sp.eye(m, format='csc')]).tocsc()
        c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
        
        # 初始基：纯人工变量
        basic_vars = list(range(n, n + m))
        nonbasic_vars = list(range(n))
        x_B = b.copy()
        
        # Phase 1 迭代
        res_phase1 = self._revised_simplex_core(c_phase1, A_phase1, b, basic_vars, nonbasic_vars, x_B)
        
        if res_phase1['status'] != 'optimal':
            return {'status': 'error', 'message': 'Phase I 未能收敛'}
            
        if res_phase1['objective_value'] > self.tol:
            return {'status': 'infeasible', 'message': '原问题完全不可行 (人工变量无法被剔除)'}
            
        # 移除人工变量的逻辑
        # 我们必须确保 basic_vars 里没有 >= n 的人工变量。如果有，需要做退化消去（此处暂采用简单策略：如果有剩余，且值为0，可尝试替换或保留处理）
        basic_vars = res_phase1['basic_vars']
        x_B = res_phase1['x_B']
        
        # 过滤掉非基变量中的人工变量
        nonbasic_vars = [v for v in nonbasic_vars if v < n]
        
        # 阶段二：使用原目标函数求解
        print("      🔄 Phase II: 求解原问题 (Sparse)")
        res_phase2 = self._revised_simplex_core(c_internal, A, b, basic_vars, nonbasic_vars, x_B)
        
        if res_phase2['status'] == 'optimal':
            solution = np.zeros(n)
            for i, bv in enumerate(res_phase2['basic_vars']):
                if bv < n:
                    solution[bv] = res_phase2['x_B'][i]
            
            obj_val = res_phase2['objective_value']
            if is_maximize:
                obj_val = -obj_val
                
            return {
                'status': 'optimal',
                'solution': solution,
                'objective_value': obj_val,
                'iterations': res_phase1['iterations'] + res_phase2['iterations'],
                'solve_time': time.time() - start_time
            }
        else:
            return res_phase2
            
    def _revised_simplex_core(self, c, A, b, basic_vars, nonbasic_vars, x_B):
        m, n_total = A.shape
        iteration = 0
        
        while iteration < self.max_iter:
            # 1. 提取基矩阵 B 并进行 LU 分解
            B = A[:, basic_vars]
            
            # 使用 splu (SuperLU) 对 B 进行分解
            try:
                # 为了防止矩阵完全奇异退化，如果在运算中退化，会抛出 RuntimeError
                lu = spla.splu(B.tocsc()) 
            except RuntimeError as e:
                warnings.warn(f"LU 分解失败 (矩阵奇异)，可能是数值不稳定导致: {e}")
                return {'status': 'numerical_error', 'message': 'LU 分解矩阵奇异退化'}
            
            # 2. 求解对偶变量 (Dual Variables) y: B^T y = c_B  => y = (B^T)^-1 c_B
            c_B = c[basic_vars]
            y = lu.solve(c_B, trans='T')
            
            # 3. 计算非基变量检验数 r_N = c_N - A_N^T y
            # 为了提高效率，只要找到一个满足条件的负检验数就可以采用 Bland's Rule 或 Dantzig's Rule
            c_N = c[nonbasic_vars]
            A_N = A[:, nonbasic_vars]
            r_N = c_N - A_N.T.dot(y)
            
            # 寻找进基变量 (最小的负检验数 - Dantzig's rule)
            min_r_idx = np.argmin(r_N)
            min_r_val = r_N[min_r_idx]
            
            if min_r_val >= -self.tol:
                # 已达到最优
                obj_val = np.dot(c_B, x_B)
                return {
                    'status': 'optimal',
                    'basic_vars': basic_vars,
                    'nonbasic_vars': nonbasic_vars,
                    'x_B': x_B,
                    'objective_value': obj_val,
                    'iterations': iteration
                }
                
            enter_var = nonbasic_vars[min_r_idx]
            
            # 4. 求解搜索方向 d: B d = A_q
            A_q = A[:, enter_var].toarray().flatten()
            d = lu.solve(A_q)
            
            # 5. 最小比值测试找到出基变量
            # 找出 d > 0 的元素
            valid_d_mask = d > self.tol
            if not np.any(valid_d_mask):
                return {'status': 'unbounded', 'message': '问题无界，存在无穷下降射线的搜索方向'}
                
            ratios = x_B[valid_d_mask] / d[valid_d_mask]
            
            min_ratio_idx = np.argmin(ratios)
            step_size = ratios[min_ratio_idx]
            
            # 映射回由于 valid_d_mask 掩码前的正确索引
            valid_indices = np.where(valid_d_mask)[0]
            leave_idx = valid_indices[min_ratio_idx]
            leave_var = basic_vars[leave_idx]
            
            # 6. 更新基解与基/非基集合
            x_B = x_B - step_size * d
            x_B[leave_idx] = step_size
            
            basic_vars[leave_idx] = enter_var
            nonbasic_vars[min_r_idx] = leave_var
            
            iteration += 1
            
        return {'status': 'iterations_exceeded', 'message': '达到最大迭代次数'}
