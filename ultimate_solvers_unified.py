#!/usr/bin/env python3
"""
终极求解器库统一接口
提供所有求解器的统一入口和便捷函数
"""

import sys
import io

# 设置标准输出为 UTF-8 以避免 Windows 下的 gbk 编码错误
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 版本信息
__version__ = "1.0.0"
__author__ = "Ultimate Solver Team"
__description__ = "终极线性规划和层次分析法求解器库"

# 导入所有求解器
try:
    from ultimate_lp_solver import UltimateLPSolver
    ULTIMATE_LP_AVAILABLE = True
except ImportError:
    ULTIMATE_LP_AVAILABLE = False

try:
    from ultimate_ahp_solver import UltimateAHPSolver, solve_ahp_problem
    ULTIMATE_AHP_AVAILABLE = True
except ImportError:
    ULTIMATE_AHP_AVAILABLE = False

try:
    from ultimate_decision_solver import UltimateDecisionSolver
    ULTIMATE_DECISION_AVAILABLE = True
except ImportError:
    ULTIMATE_DECISION_AVAILABLE = False

# 导入文件读取器
try:
    from mps_reader import read_mps
    from lp_reader import read_lp
    FILE_READERS_AVAILABLE = True
except ImportError:
    FILE_READERS_AVAILABLE = False

# 导入CuPy用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def get_solver_info():
    """获取求解器库信息"""
    return {
        'version': __version__,
        'description': __description__,
        'available_solvers': {
            'linear_programming': ULTIMATE_LP_AVAILABLE,
            'ahp': ULTIMATE_AHP_AVAILABLE,
            'decision_integration': ULTIMATE_DECISION_AVAILABLE,
            'gpu_acceleration': CUPY_AVAILABLE
        },
        'dependencies': {
            'numpy': True,  # 基本依赖
            'cupy': CUPY_AVAILABLE,
            'pulp': True  # 可选依赖
        }
    }


def create_solver(solver_type: str = 'auto', verbose_options: dict = None, **kwargs):
    """
    创建求解器实例的工厂函数
    
    参数:
        solver_type: 求解器类型 ('lp', 'ahp', 'decision', 'auto')
        verbose_options: 详细日志控制选项字典
        **kwargs: 求解器初始化参数
        
    返回:
        求解器实例
    """
    if verbose_options is not None:
        kwargs['verbose_options'] = verbose_options
    
    solver_type = solver_type.lower()
    
    # 自动检测最优求解器
    if solver_type == 'auto':
        # Default auto hierarchy is application specific
        if ULTIMATE_DECISION_AVAILABLE:
            solver_type = 'decision'
        elif ULTIMATE_AHP_AVAILABLE:
            solver_type = 'ahp'
        elif ULTIMATE_LP_AVAILABLE:
            solver_type = 'lp'
        else:
            raise ImportError("没有可用的求解器")
    
    # 获取 kwargs 中的求解器底层算法配置
    lp_core = kwargs.get('solver', 'auto') 
    
    # 创建相应的求解器
    if solver_type == 'lp':
        if not ULTIMATE_LP_AVAILABLE:
            raise ImportError("线性规划求解器不可用")
        return UltimateLPSolver(**kwargs)
    
    elif solver_type == 'ahp':
        if not ULTIMATE_AHP_AVAILABLE:
            raise ImportError("AHP求解器不可用")
        return UltimateAHPSolver(**kwargs)
    
    elif solver_type == 'decision':
        if not ULTIMATE_DECISION_AVAILABLE:
            raise ImportError("决策分析求解器不可用")
        return UltimateDecisionSolver(**kwargs)
    
    else:
        raise ValueError(f"不支持的求解器类型: {solver_type}")


def quick_solve_lp(objective: dict, constraints: list, variables: list, use_gpu: bool = None, use_npu: bool = False, npu_cores: int = 2, method: str = 'auto', verbose_options: dict = None):
    """
    快速求解线性规划问题
    
    参数:
        objective: 目标函数定义
        constraints: 约束条件
        variables: 变量定义
        use_gpu: 是否使用GPU加速（None为自动选择）
        use_npu: 是否使用NPU Nano-slicing
        npu_cores: NPU虚拟核心数
        method: 底层求解器算法 ('pdhg', 'pulp', 'builtin', 'auto')
        verbose_options: 详细日志控制选项字典
        
    返回:
        求解结果
    """
    # 检查是否包含非线性 (NLP) 特征
    is_nlp = objective.get('is_nlp', False) or any(c.get('is_nonlinear', False) for c in constraints)
    if is_nlp:
        try:
            from ultimate_nlp_solver import UltimateNLPSolver
            solver = UltimateNLPSolver(verbose_options=verbose_options)
            return solver.solve(objective, constraints, variables)
        except ImportError:
            raise ImportError("存在非线性函数 (如 @EXP, @LOG)，但无法加载 scipy 或 ultimate_nlp_solver。")
            
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE
    
    solver = create_solver('lp', verbose_options=verbose_options, use_gpu=use_gpu, use_npu=use_npu, npu_cores=npu_cores, solver=method)
    return solver.solve(objective, constraints, variables)


def quick_solve_ahp(criteria_comparisons: list, alternative_comparisons: dict, use_gpu: bool = None):
    """
    快速求解AHP问题
    
    参数:
        criteria_comparisons: 准则层比较矩阵
        alternative_comparisons: 方案层比较矩阵
        use_gpu: 是否使用GPU加速
        
    返回:
        AHP分析结果
    """
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE
    
    solver = create_solver('ahp', use_gpu=use_gpu)
    return solver.solve({
        'criteria_comparisons': criteria_comparisons,
        'alternative_comparisons': alternative_comparisons
    })


def quick_solve_decision(ahp_problem: dict, lp_constraints: list, variable_bounds: list, use_gpu: bool = None):
    """
    快速求解决策问题（AHP + LP）
    
    参数:
        ahp_problem: AHP问题定义
        lp_constraints: 线性规划约束
        variable_bounds: 变量边界
        use_gpu: 是否使用GPU加速
        
    返回:
        决策分析结果
    """
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE
    
    solver = create_solver('decision', use_gpu=use_gpu)
    return solver.solve_ahp_weighted_lp(ahp_problem, lp_constraints, variable_bounds)


def solve_from_file(filepath: str, use_gpu: bool = None, use_npu: bool = False, npu_cores: int = 2, method: str = 'auto'):
    """
    从标准数学规划文件 (.mps / .lp) 读取并求解模型。

    参数:
        filepath: 文件路径
        use_gpu: 是否使用GPU加速
        use_npu: 是否使用NPU边缘调度
        method: 求解器算法 ('pdhg', 'pulp', 'builtin', 'auto')
    
    返回:
        求解结果字典
    """
    if not FILE_READERS_AVAILABLE:
        raise ImportError("无法加载文件读取器模块(mps_reader/lp_reader)。请确保这些文件存在。")
        
    import os
    ext = os.path.splitext(filepath)[1].lower()
    
    print(f"📂 正在解析模型文件: {filepath} ...")
    if ext == '.mps':
        problem = read_mps(filepath)
    elif ext == '.lp':
        problem = read_lp(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}。目前仅支持 .mps 和 .lp")
        
    print(f"📦 文件解析完成! 检测到 {len(problem['variables'])} 个变量, {len(problem['constraints'])} 个约束。开始求解...")
    
    return quick_solve_lp(
        problem['objective'],
        problem['constraints'],
        problem['variables'],
        use_gpu=use_gpu,
        use_npu=use_npu,
        npu_cores=npu_cores,
        method=method
    )


def benchmark_solvers(problem_size: str = 'medium'):
    """
    求解器性能基准测试
    
    参数:
        problem_size: 问题规模 ('small', 'medium', 'large')
        
    返回:
        基准测试结果
    """
    import time
    
    results = {
        'problem_size': problem_size,
        'test_time': time.time(),
        'cpu_results': {},
        'gpu_results': {},
        'solver_info': get_solver_info()
    }
    
    if not ULTIMATE_AHP_AVAILABLE:
        return {'error': 'AHP求解器不可用'}
    
    # 根据问题规模设置参数
    size_params = {
        'small': {'n_criteria': 3, 'n_alternatives': 4},
        'medium': {'n_criteria': 5, 'n_alternatives': 8},
        'large': {'n_criteria': 10, 'n_alternatives': 15}
    }
    
    if problem_size not in size_params:
        problem_size = 'medium'
    
    params = size_params[problem_size]
    
    # 生成测试问题
    import numpy as np
    np.random.seed(42)
    
    criteria_comparisons = []
    for i in range(params['n_criteria'] - 1):
        row = []
        for j in range(i + 1, params['n_criteria']):
            value = np.random.randint(1, 6)
            row.append(value)
        criteria_comparisons.append(row)
    
    alternative_comparisons = {}
    for i in range(params['n_criteria']):
        criterion_name = f'准则{i+1}'
        comparisons = []
        for j in range(params['n_alternatives'] - 1):
            row = []
            for k in range(j + 1, params['n_alternatives']):
                value = np.random.randint(1, 6)
                row.append(value)
            comparisons.append(row)
        alternative_comparisons[criterion_name] = comparisons
    
    problem = {
        'criteria_comparisons': criteria_comparisons,
        'alternative_comparisons': alternative_comparisons
    }
    
    # CPU测试
    print(f"🖥️  CPU基准测试...")
    cpu_solver = create_solver('ahp', use_gpu=False)
    start_time = time.time()
    cpu_result = cpu_solver.solve(problem)
    cpu_time = time.time() - start_time
    results['cpu_results'] = {
        'solve_time': cpu_time,
        'consistency_cr': cpu_result.get('consistency_analysis', {}).get('overall_cr', 0),
        'solver_status': 'success' if cpu_result.get('status') != 'error' else 'error'
    }
    
    # GPU测试
    if CUPY_AVAILABLE:
        print(f"🚀 GPU基准测试...")
        gpu_solver = create_solver('ahp', use_gpu=True)
        start_time = time.time()
        gpu_result = gpu_solver.solve(problem)
        gpu_time = time.time() - start_time
        results['gpu_results'] = {
            'solve_time': gpu_time,
            'consistency_cr': gpu_result.get('consistency_analysis', {}).get('overall_cr', 0),
            'solver_status': 'success' if gpu_result.get('status') != 'error' else 'error'
        }
        
        # 计算加速比
        if gpu_time > 0:
            results['speedup'] = cpu_time / gpu_time
    else:
        results['gpu_results'] = {'error': 'CuPy不可用，无法进行GPU测试'}
        results['speedup'] = None
    
    return results


# 导出所有公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    
    # 求解器类
    'UltimateLPSolver',
    'UltimateAHPSolver', 
    'UltimateDecisionSolver',
    
    # 工厂函数
    'create_solver',
    'get_solver_info',
    
    # 快速求解函数
    'quick_solve_lp',
    'quick_solve_ahp',
    'quick_solve_decision',
    'solve_from_file',
    
    # 基准测试
    'benchmark_solvers',
    
    # 工具函数
    'solve_ahp_problem'
]


def print_banner():
    """打印求解器库横幅信息"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                   终极求解器库 v{__version__:<10}                     ║
║        线性规划 + 一阶PDHG调度 + GPU/NPU加速 + 决策集成      ║
╠══════════════════════════════════════════════════════════════╣
║  📊 线性规划求解器: {'✅ 启用' if ULTIMATE_LP_AVAILABLE else '❌ 禁用':<15}                ║
║  🚀 PDHG一阶算法引擎: ✅ 启用                                  ║
║  📱 边缘设备NPU切片: ✅ 启用                                  ║
║  🎯 AHP层次分析法: {'✅ 启用' if ULTIMATE_AHP_AVAILABLE else '❌ 禁用':<15}                ║
║  🔗 决策分析集成: {'✅ 启用' if ULTIMATE_DECISION_AVAILABLE else '❌ 禁用':<15}                ║
║  🚀 GPU CuPy加速: {'✅ 启用' if CUPY_AVAILABLE else '❌ 禁用':<15}                ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    print_banner()
    print("🔍 求解器库状态检查:")
    info = get_solver_info()
    for solver, available in info['available_solvers'].items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"   {solver}: {status}")
    
    print("\n🏃‍♂️ 运行 LP PDHG vs 传统算法 快速效能测试...")
    
    test_objective = {
        'type': 'maximize',
        'coeffs': [40, 30] 
    }
    test_constraints = [
        {'type': '<=', 'coeffs': [2, 1], 'rhs': 100}, 
        {'type': '<=', 'coeffs': [1, 2], 'rhs': 80},  
    ]
    test_variables = [
        {'name': 'x1', 'type': 'nonneg'},
        {'name': 'x2', 'type': 'nonneg'}
    ]
    
    print("\n=== 1. 使用内置单纯形法求解 ===")
    res1 = quick_solve_lp(test_objective, test_constraints, test_variables, method='builtin')
    print(f"解: {res1.get('solution')}, 耗时: {res1.get('solve_time', 0):.5f}s\n")
    
    print("=== 2. 使用极速 PDHG 阶梯法求解 (适合 CPU/NPU/GPU) ===")
    res2 = quick_solve_lp(test_objective, test_constraints, test_variables, method='pdhg')
    print(f"解: {res2.get('solution')}, 耗时: {res2.get('solve_time', 0):.5f}s\n")
    
    print("=== 3. 启用 NPU 边缘设备调度器求解 ===")
    res3 = quick_solve_lp(test_objective, test_constraints, test_variables, method='pdhg', use_npu=True, npu_cores=4)
    print(f"解: {res3.get('solution')}, 耗时: {res3.get('solve_time', 0):.5f}s\n")
    
    print("\n🎉 求解器库检查完成!")