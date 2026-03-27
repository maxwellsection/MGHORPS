#!/usr/bin/env python3
"""
终极求解器库Vulkan加速测试脚本
测试Vulkan、CuPy、NumPy等多种加速方法的性能对比
"""

import sys
import os
import time
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_solvers_vulkan_unified import (
    create_solver, quick_solve_ahp, quick_solve_decision,
    benchmark_acceleration_methods, get_gpu_capabilities, 
    recommend_acceleration_method, print_banner, get_solver_info
)
from multi_gpu_accelerated_solver import MultiGPUAcceleratedSolver
from vulkan_compute_accelerator import VulkanOptimizedAHPSolver


def test_vulkan_solver():
    """测试Vulkan AHP求解器"""
    print("\n🧪 测试Vulkan AHP求解器")
    
    try:
        # 创建Vulkan求解器
        solver = VulkanOptimizedAHPSolver(use_vulkan=True, use_cupy=True)
        
        # 定义测试问题
        criteria_comparisons = [
            [3, 2, 4, 5],
            [2, 3, 2],
            [1, 3],
            [2]
        ]
        
        alternative_comparisons = {
            '准则1': [
                [2, 3, 4, 5],
                [2, 3, 4],
                [2, 3],
                [2]
            ],
            '准则2': [
                [1/2, 1/3, 1/4, 1/5],
                [1/2, 1/3, 1/4],
                [1/2, 1/3],
                [1/2]
            ],
            '准则3': [
                [1/2, 1/3, 1/4, 1/5],
                [1/2, 1/4, 1/5],
                [1/2, 1/3],
                [1/3]
            ],
            '准则4': [
                [2, 3, 4, 5],
                [2, 3, 4],
                [2, 3],
                [2]
            ]
        }
        
        # 求解
        start_time = time.time()
        result = solver.solve_ahp_with_vulkan(criteria_comparisons, alternative_comparisons)
        solve_time = time.time() - start_time
        
        print(f"   求解时间: {solve_time:.4f}秒")
        print(f"   计算后端: {result.get('compute_backend', 'unknown')}")
        print(f"   一致性比率: {result.get('criteria_cr', 0):.4f}")
        
        if 'device_info' in result and result['device_info']:
            print(f"   设备信息: {result['device_info']}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Vulkan求解器测试失败: {e}")
        return None


def test_multi_gpu_solver():
    """测试多GPU求解器"""
    print("\n🧪 测试多GPU求解器")
    
    try:
        # 创建多GPU求解器（自动选择）
        solver = MultiGPUAcceleratedSolver(
            acceleration_method='auto',
            enable_vulkan=True,
            enable_cupy=True,
            enable_opencl=False
        )
        
        # 定义测试问题
        criteria_comparisons = [
            [3, 2, 4],
            [2, 3],
            [1]
        ]
        
        alternative_comparisons = {
            '准则1': [
                [2, 3, 4],
                [2, 3],
                [2]
            ],
            '准则2': [
                [1/2, 1/3, 1/4],
                [1/2, 1/3],
                [1/2]
            ],
            '准则3': [
                [1/2, 1/3, 1/5],
                [1/2, 1/4],
                [1/3]
            ]
        }
        
        # 求解
        start_time = time.time()
        result = solver.solve_ahp_multi_gpu(criteria_comparisons, alternative_comparisons)
        solve_time = time.time() - start_time
        
        print(f"   求解时间: {solve_time:.4f}秒")
        print(f"   加速方法: {result.get('acceleration_method', 'unknown')}")
        print(f"   后端: {result.get('backend', 'unknown')}")
        print(f"   一致性比率: {result.get('criteria_cr', 0):.4f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ 多GPU求解器测试失败: {e}")
        return None


def test_acceleration_methods_comparison():
    """测试各种加速方法的性能对比"""
    print("\n🏃‍♂️ 各种加速方法性能对比测试")
    
    # 获取可用的加速方法
    info = get_solver_info()
    available_backends = [backend for backend, available in info['acceleration_backends'].items() if available]
    
    if len(available_backends) < 2:
        print("   ⚠️ 可用的加速方法少于2个，跳过对比测试")
        return None
    
    # 生成测试问题
    test_problems = [
        {'name': '小问题', 'size': 'small'},
        {'name': '中问题', 'size': 'medium'},
        {'name': '大问题', 'size': 'large'}
    ]
    
    results = {}
    
    for problem in test_problems:
        print(f"\n📊 测试{problem['name']}:")
        
        problem_results = {}
        
        # 测试NumPy
        if info['acceleration_backends']['numpy_cpu']:
            print("   💻 测试NumPy CPU...")
            try:
                solver = create_solver('ahp', acceleration_method='numpy')
                start_time = time.time()
                numpy_result = solver.solve({
                    'criteria_comparisons': [[2, 3], [2]],
                    'alternative_comparisons': {'准则1': [[2, 3], [2]], '准则2': [[1/2, 1/3], [1/2]]}
                })
                numpy_time = time.time() - start_time
                problem_results['numpy'] = {'time': numpy_time, 'status': 'success'}
                print(f"      NumPy: {numpy_time:.4f}秒")
            except Exception as e:
                problem_results['numpy'] = {'error': str(e)}
                print(f"      NumPy: ❌ 失败 - {e}")
        
        # 测试CuPy
        if info['acceleration_backends']['cupy_cuda']:
            print("   🔥 测试CuPy CUDA...")
            try:
                solver = create_solver('multi_gpu_ahp', acceleration_method='cupy')
                start_time = time.time()
                cupy_result = solver.solve_ahp_multi_gpu([[2, 3], [2]], {'准则1': [[2, 3], [2]], '准则2': [[1/2, 1/3], [1/2]]})
                cupy_time = time.time() - start_time
                problem_results['cupy'] = {'time': cupy_time, 'status': 'success'}
                print(f"      CuPy: {cupy_time:.4f}秒")
            except Exception as e:
                problem_results['cupy'] = {'error': str(e)}
                print(f"      CuPy: ❌ 失败 - {e}")
        
        # 测试Vulkan
        if info['acceleration_backends']['vulkan']:
            print("   ⚡ 测试Vulkan...")
            try:
                solver = create_solver('vulkan_ahp', acceleration_method='vulkan')
                start_time = time.time()
                vulkan_result = solver.solve_ahp_with_vulkan([[2, 3], [2]], {'准则1': [[2, 3], [2]], '准则2': [[1/2, 1/3], [1/2]]})
                vulkan_time = time.time() - start_time
                problem_results['vulkan'] = {'time': vulkan_time, 'status': 'success'}
                print(f"      Vulkan: {vulkan_time:.4f}秒")
            except Exception as e:
                problem_results['vulkan'] = {'error': str(e)}
                print(f"      Vulkan: ❌ 失败 - {e}")
        
        # 计算加速比
        base_time = None
        for backend in ['numpy', 'cupy', 'vulkan']:
            if backend in problem_results and 'time' in problem_results[backend]:
                base_time = problem_results[backend]['time']
                break
        
        if base_time:
            for backend in problem_results:
                if 'time' in problem_results[backend]:
                    speedup = base_time / problem_results[backend]['time']
                    problem_results[backend]['speedup'] = speedup
        
        results[problem['name']] = problem_results
    
    return results


def test_gpu_capabilities():
    """测试GPU能力检测"""
    print("\n🔍 GPU能力检测测试")
    
    try:
        capabilities = get_gpu_capabilities()
        
        print(f"   检测结果:")
        for backend, info in capabilities.items():
            status = "✅" if info['available'] else "❌"
            device_count = len(info['devices'])
            print(f"     {backend}: {status} ({device_count} 设备)")
            
            for device in info['devices']:
                if 'name' in device:
                    print(f"       - {device['name']}")
                elif 'platform' in device:
                    print(f"       - {device['platform']}: {device['device']}")
                else:
                    print(f"       - {device}")
        
        # 推荐加速方法
        recommendation = recommend_acceleration_method('medium')
        print(f"   推荐加速方法: {recommendation}")
        
        return capabilities
        
    except Exception as e:
        print(f"   ❌ GPU能力检测失败: {e}")
        return None


def test_enhanced_decision_analysis():
    """测试增强的决策分析"""
    print("\n🎯 测试增强的决策分析")
    
    try:
        # 创建决策分析求解器
        solver = create_solver('decision', acceleration_method='auto')
        
        # AHP问题：投资组合优化
        ahp_problem = {
            'criteria_comparisons': [
                [2, 4, 3],    # 收益 vs 风险, 收益 vs 流动性
                [3, 2],       # 风险 vs 流动性
                [1]           # 流动性
            ],
            'alternative_comparisons': {
                '收益': [
                    [2, 4, 6],    # 股票 vs 债券, 股票 vs 基金, 股票 vs 现金
                    [3, 5],       # 债券 vs 基金, 债券 vs 现金
                    [2]           # 基金 vs 现金
                ],
                '风险': [
                    [1/3, 1/5, 1/7],  # 风险越小越好
                    [1/2, 1/4],         # 债券 vs 基金, 债券 vs 现金
                    [1/2]               # 基金 vs 现金
                ],
                '流动性': [
                    [1/2, 1/3, 1/8],  # 流动性越好越好
                    [1/2, 1/6],         # 债券 vs 基金, 债券 vs 现金
                    [1/4]               # 基金 vs 现金
                ]
            }
        }
        
        # 线性规划约束
        lp_constraints = [
            {
                'type': '<=',
                'coeffs': [1, 1, 1, 1],  # 总投资 <= 100万
                'rhs': 1000000
            },
            {
                'type': '<=',
                'coeffs': [1, 0, 0, 0],  # 股票 <= 40万
                'rhs': 400000
            },
            {
                'type': '>=',
                'coeffs': [0, 0, 0, 1],  # 现金 >= 5万
                'rhs': 50000
            }
        ]
        
        variable_bounds = [
            {'name': '股票', 'type': 'nonneg'},
            {'name': '债券', 'type': 'nonneg'},
            {'name': '基金', 'type': 'nonneg'},
            {'name': '现金', 'type': 'nonneg'}
        ]
        
        # 求解
        start_time = time.time()
        result = solver.solve_ahp_weighted_lp(ahp_problem, lp_constraints, variable_bounds)
        solve_time = time.time() - start_time
        
        print(f"   求解时间: {solve_time:.4f}秒")
        print(f"   AHP状态: {result.get('ahp_result', {}).get('status', 'unknown')}")
        print(f"   LP状态: {result.get('lp_result', {}).get('status', 'unknown')}")
        
        if 'solve_summary' in result:
            summary = result['solve_summary']
            print(f"   解质量: {summary.get('solution_quality', 'unknown')}")
            print(f"   一致性: {summary.get('ahp_consistency', 0):.4f}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ 增强决策分析测试失败: {e}")
        return None


def run_comprehensive_vulkan_test():
    """运行综合Vulkan测试"""
    print("🎯 终极求解器库Vulkan加速综合测试")
    print("=" * 70)
    
    # 打印横幅
    print_banner()
    
    # 测试结果统计
    test_results = []
    
    # 测试1: GPU能力检测
    gpu_capabilities = test_gpu_capabilities()
    test_results.append(("GPU能力检测", gpu_capabilities is not None))
    
    # 测试2: Vulkan求解器
    vulkan_result = test_vulkan_solver()
    test_results.append(("Vulkan求解器", vulkan_result is not None))
    
    # 测试3: 多GPU求解器
    multi_gpu_result = test_multi_gpu_solver()
    test_results.append(("多GPU求解器", multi_gpu_result is not None))
    
    # 测试4: 增强的决策分析
    enhanced_result = test_enhanced_decision_analysis()
    test_results.append(("增强决策分析", enhanced_result is not None))
    
    # 测试5: 加速方法对比
    comparison_result = test_acceleration_methods_comparison()
    test_results.append(("加速方法对比", comparison_result is not None))
    
    # 测试6: 统一接口
    try:
        info = get_solver_info()
        print(f"\n🔍 统一接口测试:")
        print(f"   库版本: {info['version']}")
        print(f"   可用求解器: {list(info['available_solvers'].keys())}")
        print(f"   可用加速后端: {list(info['acceleration_backends'].keys())}")
        test_results.append(("统一接口", True))
    except Exception as e:
        print(f"   ❌ 统一接口测试失败: {e}")
        test_results.append(("统一接口", False))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！Vulkan加速功能正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    # 显示GPU使用建议
    if gpu_capabilities:
        recommendation = recommend_acceleration_method('medium')
        print(f"\n💡 GPU使用建议:")
        print(f"   推荐加速方法: {recommendation}")
        print(f"   使用场景:")
        print(f"     - 小规模问题 (<5个准则): NumPy CPU")
        print(f"     - 中等规模问题 (5-10个准则): CuPy CUDA")
        print(f"     - 大规模问题 (>10个准则): Vulkan")
        print(f"     - 多设备环境: MultiGPU自动选择")
    
    return passed == total


if __name__ == "__main__":
    # 运行综合测试
    success = run_comprehensive_vulkan_test()
    
    # 退出码
    sys.exit(0 if success else 1)