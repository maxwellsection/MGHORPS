#!/usr/bin/env python3
"""
Vulkan加速完整使用示例
展示如何使用Vulkan、CuPy、NumPy等多种加速方案
"""

import sys
import os
import time
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_solvers_vulkan_unified import (
    create_solver, quick_solve_ahp, quick_solve_decision,
    get_gpu_capabilities, recommend_acceleration_method, 
    print_banner, get_solver_info
)
from multi_gpu_accelerated_solver import MultiGPUAcceleratedSolver


def demo_vulkan_acceleration():
    """演示Vulkan加速功能"""
    print("🎬 Vulkan加速功能演示")
    
    # 1. 显示求解器库信息
    print("\n📊 求解器库信息:")
    info = get_solver_info()
    for solver, available in info['available_solvers'].items():
        status = "✅" if available else "❌"
        print(f"   {solver}: {status}")
    
    print("\n🔧 可用加速后端:")
    for backend, available in info['acceleration_backends'].items():
        status = "✅" if available else "❌"
        print(f"   {backend}: {status}")
    
    # 2. GPU能力检测
    print("\n🔍 GPU能力检测:")
    capabilities = get_gpu_capabilities()
    for backend, info in capabilities.items():
        device_count = len(info['devices'])
        print(f"   {backend}: {device_count} 设备")
    
    # 3. 推荐加速方法
    print("\n💡 加速方法推荐:")
    for size in ['small', 'medium', 'large']:
        recommendation = recommend_acceleration_method(size)
        print(f"   {size}问题: {recommendation}")
    
    # 4. 使用不同加速方法求解
    print("\n🚀 使用不同加速方法求解AHP问题:")
    
    # 定义测试问题
    criteria_comparisons = [
        [3, 2, 4, 5],   # 准则1 vs 其他
        [2, 3, 2],      # 准则2 vs 其他
        [1, 3],         # 准则3 vs 其他
        [2]              # 准则4 vs 其他
    ]
    
    alternative_comparisons = {
        '收益': [
            [2, 4, 6, 8],   # 股票 vs 债券, 股票 vs 基金, 股票 vs 现金, 股票 vs 期货
            [3, 5, 7],       # 债券 vs 基金, 债券 vs 现金, 债券 vs 期货
            [2, 4],          # 基金 vs 现金, 基金 vs 期货
            [2]              # 现金 vs 期货
        ],
        '风险': [
            [1/3, 1/5, 1/7, 1/9],  # 风险越小越好
            [1/2, 1/4, 1/6],
            [1/2, 1/4],
            [1/2]
        ],
        '流动性': [
            [1/2, 1/3, 1/8, 1/9],  # 流动性越好越好
            [1/2, 1/6, 1/7],
            [1/4, 1/5],
            [1/3]
        ],
        '社会责任': [
            [2, 1, 4, 6],   # 社会责任越高越好
            [1/2, 3, 5],
            [2, 4],
            [3]
        ]
    }
    
    # 测试各种方法
    methods = ['numpy', 'cupy']
    
    for method in methods:
        print(f"\n📈 使用{method}方法:")
        try:
            start_time = time.time()
            
            if method == 'numpy':
                solver = create_solver('ahp', acceleration_method='numpy')
                result = solver.solve({
                    'criteria_comparisons': criteria_comparisons,
                    'alternative_comparisons': alternative_comparisons
                })
            else:  # cupy
                solver = create_solver('multi_gpu_ahp', acceleration_method='cupy')
                result = solver.solve_ahp_multi_gpu(criteria_comparisons, alternative_comparisons)
            
            solve_time = time.time() - start_time
            
            print(f"   求解时间: {solve_time:.4f}秒")
            print(f"   状态: {result.get('status', 'success')}")
            print(f"   一致性: {result.get('criteria_cr', 0):.4f}")
            
            if 'criteria_weights' in result:
                weights = result['criteria_weights']
                if hasattr(weights, 'get'):
                    weights = weights.get()
                print(f"   准则权重: {weights[:3]}...")  # 只显示前3个
            
        except Exception as e:
            print(f"   ❌ {method}方法失败: {e}")


def demo_multi_gpu_decision():
    """演示多GPU决策分析"""
    print("\n🎯 多GPU决策分析演示")
    
    # 投资组合优化问题
    print("\n💼 投资组合优化问题:")
    
    # AHP问题定义
    ahp_problem = {
        'criteria_comparisons': [
            [2, 4, 3, 5],   # 收益 vs 风险, 收益 vs 流动性, 收益 vs 稳定性
            [3, 2, 3],      # 风险 vs 流动性, 风险 vs 稳定性
            [1, 2],          # 流动性 vs 稳定性
            [2]              # 稳定性比较
        ],
        'alternative_comparisons': {
            '收益': [
                [2, 4, 6, 8],    # 股票 vs 债券, 股票 vs 基金, 股票 vs 现金, 股票 vs 期货
                [3, 5, 7],       # 债券 vs 基金, 债券 vs 现金, 债券 vs 期货
                [2, 4],          # 基金 vs 现金, 基金 vs 期货
                [2]              # 现金 vs 期货
            ],
            '风险': [
                [1/3, 1/5, 1/7, 1/9],  # 风险越小越好
                [1/2, 1/4, 1/6],
                [1/2, 1/4],
                [1/2]
            ],
            '流动性': [
                [1/2, 1/3, 1/8, 1/9],  # 流动性越好越好
                [1/2, 1/6, 1/7],
                [1/4, 1/5],
                [1/3]
            ],
            '稳定性': [
                [2, 3, 4, 5],    # 稳定性越高越好
                [2, 3, 4],
                [2, 3],
                [2]
            ]
        }
    }
    
    # 线性规划约束
    lp_constraints = [
        {
            'type': '<=',
            'coeffs': [1, 1, 1, 1, 1],  # 总投资 <= 100万
            'rhs': 1000000
        },
        {
            'type': '<=',
            'coeffs': [1, 0, 0, 0, 0],  # 股票 <= 30万
            'rhs': 300000
        },
        {
            'type': '<=',
            'coeffs': [0, 1, 0, 0, 0],  # 债券 <= 40万
            'rhs': 400000
        },
        {
            'type': '<=',
            'coeffs': [0, 0, 1, 0, 0],  # 基金 <= 20万
            'rhs': 200000
        },
        {
            'type': '>=',
            'coeffs': [0, 0, 0, 1, 0],  # 现金 >= 5万
            'rhs': 50000
        },
        {
            'type': '>=',
            'coeffs': [0, 0, 0, 0, 1],  # 期货 >= 2万
            'rhs': 20000
        }
    ]
    
    variable_bounds = [
        {'name': '股票', 'type': 'nonneg'},
        {'name': '债券', 'type': 'nonneg'},
        {'name': '基金', 'type': 'nonneg'},
        {'name': '现金', 'type': 'nonneg'},
        {'name': '期货', 'type': 'nonneg'}
    ]
    
    # 求解
    print("   🔄 正在求解...")
    start_time = time.time()
    
    try:
        result = create_solver('decision', acceleration_method='auto').solve_ahp_weighted_lp(
            ahp_problem, lp_constraints, variable_bounds
        )
        
        solve_time = time.time() - start_time
        
        print(f"   ✅ 求解完成! ({solve_time:.4f}秒)")
        
        # 显示结果摘要
        if 'solve_summary' in result:
            summary = result['solve_summary']
            print(f"   📊 解质量: {summary.get('solution_quality', 'unknown')}")
            print(f"   📈 一致性: {summary.get('ahp_consistency', 0):.4f}")
            print(f"   🎯 LP状态: {summary.get('lp_feasibility', False)}")
        
        if 'lp_result' in result and result['lp_result'].get('solution'):
            solution = result['lp_result']['solution']
            if hasattr(solution, 'get'):
                solution = solution.get()
            print(f"   💰 最优投资组合:")
            for i, amount in enumerate(solution[:4]):  # 显示前4个投资
                print(f"      {variable_bounds[i]['name']}: {amount:,.0f}")
        
    except Exception as e:
        print(f"   ❌ 求解失败: {e}")


def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 性能对比演示")
    
    # 生成不同规模的问题
    problems = [
        {'name': '小问题', 'n_criteria': 3, 'n_alternatives': 4},
        {'name': '中问题', 'n_criteria': 5, 'n_alternatives': 6},
        {'name': '大一些问题', 'n_criteria': 7, 'n_alternatives': 8}
    ]
    
    for problem in problems:
        print(f"\n📊 {problem['name']} ({problem['n_criteria']}准则, {problem['n_alternatives']}方案):")
        
        # 生成随机问题
        np.random.seed(42)
        
        criteria_comparisons = []
        for i in range(problem['n_criteria'] - 1):
            row = []
            for j in range(i + 1, problem['n_criteria']):
                value = np.random.randint(1, 4)
                row.append(value)
            criteria_comparisons.append(row)
        
        alternative_comparisons = {}
        for i in range(problem['n_criteria']):
            criterion_name = f'准则{i+1}'
            comparisons = []
            for j in range(problem['n_alternatives'] - 1):
                row = []
                for k in range(j + 1, problem['n_alternatives']):
                    value = np.random.randint(1, 4)
                    row.append(value)
                comparisons.append(row)
            alternative_comparisons[criterion_name] = comparisons
        
        # 测试NumPy
        try:
            print("   💻 NumPy CPU...")
            start_time = time.time()
            numpy_result = create_solver('ahp', acceleration_method='numpy').solve({
                'criteria_comparisons': criteria_comparisons,
                'alternative_comparisons': alternative_comparisons
            })
            numpy_time = time.time() - start_time
            print(f"      时间: {numpy_time:.4f}秒")
        except Exception as e:
            numpy_time = float('inf')
            print(f"      失败: {e}")
        
        # 测试CuPy（如果有GPU）
        try:
            print("   🔥 CuPy GPU...")
            start_time = time.time()
            cupy_result = create_solver('multi_gpu_ahp', acceleration_method='cupy').solve_ahp_multi_gpu(
                criteria_comparisons, alternative_comparisons
            )
            cupy_time = time.time() - start_time
            print(f"      时间: {cupy_time:.4f}秒")
            
            if numpy_time < float('inf'):
                speedup = numpy_time / cupy_time
                print(f"      加速比: {speedup:.2f}x")
                
        except Exception as e:
            print(f"      失败: {e}")


def demo_vulkan_simulation():
    """演示Vulkan模拟功能"""
    print("\n⚡ Vulkan加速模拟演示")
    
    print("   📋 注意: 当前环境Vulkan不可用，但系统支持回退机制")
    print("   🔄 自动回退到CuPy...")
    
    # 模拟Vulkan优化的AHP求解
    solver = MultiGPUAcceleratedSolver(
        acceleration_method='auto',
        enable_vulkan=True,
        enable_cupy=True
    )
    
    # 定义一个简单的问题
    criteria_comparisons = [
        [3, 2],
        [1]
    ]
    
    alternative_comparisons = {
        '准则1': [[2, 3], [2]],
        '准则2': [[1/2, 1/3], [1/2]]
    }
    
    print("   🧮 执行多GPU AHP求解...")
    start_time = time.time()
    
    result = solver.solve_ahp_multi_gpu(criteria_comparisons, alternative_comparisons)
    
    solve_time = time.time() - start_time
    
    print(f"   ✅ 求解完成! ({solve_time:.4f}秒)")
    print(f"   🔧 使用后端: {result.get('backend', 'unknown')}")
    print(f"   📊 加速方法: {result.get('acceleration_method', 'unknown')}")
    
    if 'criteria_weights' in result:
        weights = result['criteria_weights']
        if hasattr(weights, 'get'):
            weights = weights.get()
        print(f"   📈 准则权重: {weights}")


def main():
    """主演示函数"""
    print("🎯 终极求解器库Vulkan加速完整演示")
    print("=" * 60)
    
    # 显示横幅
    print_banner()
    
    # 运行各种演示
    demo_vulkan_acceleration()
    demo_multi_gpu_decision()
    demo_performance_comparison()
    demo_vulkan_simulation()
    
    print("\n" + "=" * 60)
    print("🎉 所有演示完成!")
    
    print("\n💡 使用提示:")
    print("   • 小规模问题: 使用NumPy CPU即可")
    print("   • 中规模问题: 建议使用CuPy GPU")
    print("   • 大规模问题: 优先选择Vulkan")
    print("   • 多设备环境: 使用auto模式自动选择")
    
    print("\n🚀 支持的加速方法:")
    print("   • numpy: NumPy CPU计算")
    print("   • cupy: CuPy GPU计算")
    print("   • vulkan: Vulkan GPU计算")
    print("   • opencl: OpenCL GPU计算")
    print("   • auto: 自动选择最佳方法")


if __name__ == "__main__":
    main()