import time
import numpy as np
import scipy.sparse as sp
from ultimate_lp_solver import UltimateLPSolver

def test_large_sparse_problem():
    print("====== 测试高维稀疏规模 LP (Sparse Revised Simplex) ======")
    np.random.seed(42)
    
    # 构建 500 个约束， 1000 个变量的稀疏矩阵
    m = 500
    n = 1000
    
    # 目标函数 (全部最小化)
    c = np.random.uniform(-1, 1, n)
    objective = {'type': 'min', 'coeffs': c.tolist()}
    
    # 约束
    constraints = []
    
    # 为了保证初始有可行解，我们构造一个已知解 x* = [1,1,...,1]，然后通过 b = Ax 推导 b
    true_x = np.ones(n)
    
    print(f"正在构建 {m}x{n} 规模且稀疏度 5% 的随机矩阵...")
    for i in range(m):
        # 每行只有 5% 非零元素
        non_zeros_indices = np.random.choice(n, size=int(n*0.05), replace=False)
        coeffs = np.zeros(n)
        coeffs[non_zeros_indices] = np.random.uniform(-2, 2, len(non_zeros_indices))
        
        # 强制满足已知可行解
        rhs = float(np.dot(coeffs, true_x))
        # 偶尔加点等式和不等式
        ctype = '='
        if i % 3 == 0:
            ctype = '<='
            rhs += np.random.uniform(0, 5) # 使不等式宽松一点
        elif i % 3 == 1:
            ctype = '>='
            rhs -= np.random.uniform(0, 5)
            
        constraints.append({
            'type': ctype,
            'coeffs': coeffs.tolist(),
            'rhs': rhs,
            'name': f'c{i}'
        })
        
    variables = [{'name': f'x{i}', 'type': 'nonneg'} for i in range(n)]
    
    print("\n【运行基准 - PuLP求解器】...")
    solver_pulp = UltimateLPSolver(solver='pulp')
    t0 = time.time()
    try:
        res_pulp = solver_pulp.solve(objective, constraints, variables)
    except Exception as e:
        print(f"PuLP failed: {e}")
        res_pulp = None
    pulp_time = time.time() - t0

    print("\n【运行修改后的稀疏基元求解器 (revised_simplex)】...")
    solver_sparse = UltimateLPSolver(solver='revised_simplex', tolerance=1e-6)
    t1 = time.time()
    try:
        res_sparse = solver_sparse.solve(objective, constraints, variables)
    except Exception as e:
        print(f"Sparse Revised failed: {e}")
        res_sparse = None
    sparse_time = time.time() - t1
    
    print("\n====== 对比报告 ======")
    
    if res_pulp:
        print(f"PuLP 基准状态: {res_pulp['status']}")
        if res_pulp.get('objective_value') is not None:
             print(f"PuLP 目标值: {res_pulp['objective_value']:.4f}")

    if res_sparse and res_sparse['status'] == 'optimal':
        print(f"Sparse Revised 求解耗时: {sparse_time:.4f} 秒, 目标值: {res_sparse['objective_value']:.4f}")
        print(f"迭代次数: {res_sparse.get('iterations', 'Unknown')}")
    else:
        print(f"Sparse Revised 求解失败: {res_sparse['status'] if res_sparse else 'Error'} - {res_sparse.get('message', '') if res_sparse else ''}")

if __name__ == '__main__':
    # 避免 numpy 输出警告
    import warnings
    warnings.filterwarnings('ignore')
    
    test_large_sparse_problem()
