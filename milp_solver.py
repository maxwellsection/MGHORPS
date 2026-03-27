#!/usr/bin/env python3
"""
终极混合整数线性规划求解器 (MILP)
通过高级分支定界 (Branch and Bound) 和原始启发式算法
将 UltimateLPSolver 的连续求解能力升级为解决工业级整数规划问题的能力。
"""

import sys
import io

# 设置标准输出为 UTF-8 以避免 Windows 下的 gbk 编码错误
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import copy
import time
import math
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional

# 尝试导入底层的 UltimateLPSolver 作为松弛求解引擎
try:
    from ultimate_lp_solver import UltimateLPSolver
    LP_SOLVER_AVAILABLE = True
except ImportError:
    LP_SOLVER_AVAILABLE = False
    print("[WARN] 无法加载 UltimateLPSolver, MILP求解器将无法工作。")

# 常量定义
INTEGER_TOLERANCE = 1e-5  # 判断一个数是否为整数的容差
INF = float('inf')

class MILPNode:
    """分支定界搜索树的节点"""
    
    def __init__(self, bounds: Dict[int, Tuple[float, float]], depth: int = 0):
        # 记录每个变量的额外边界条件 bounds: {var_index: (lower_bound, upper_bound)}
        self.bounds = bounds
        # 节点在搜索树中的深度
        self.depth = depth
        # 当前节点的松弛解及目标值
        self.relaxation_solution = None
        self.relaxation_obj = None
        self.is_feasible = False

    def __lt__(self, other):
        """优先队列比较函数 (Best-Bound First)"""
        # 注意：这里假设是最大化问题。如果是最大化，目标值大的优先。
        # 如果要处理最小化，需要在求解器主控中处理。为了通用，直接比较松弛界限。
        if self.relaxation_obj is None or other.relaxation_obj is None:
            return self.depth > other.depth # 没算过的话优先扩展深层节点（深度优先防爆内存）
        # 逆序（优先扩展界限最好的节点）。此时存储时应该是存 -obj (如果是最大化)
        return False # 实际由外层 heapq 元组来决定

class MILPBranchAndBound:
    """混合整数规划核心求解器 - 基于分支定界"""
    
    def __init__(self, tolerance: float = 1e-5, time_limit: int = 600, **lp_kwargs):
        self.tolerance = tolerance
        self.integer_tolerance = INTEGER_TOLERANCE
        self.time_limit = time_limit
        self.lp_kwargs = lp_kwargs  # 传递给底层LP求解器的参数 (如 use_gpu, solver)
        
        # 状态记录
        self.incumbent_obj = None       # 当前最优整数解的目标值
        self.incumbent_solution = None  # 当前最优整数解
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.start_time = 0
        
        # 伪代价（Pseudo-cost）计分板
        self.pseudocost_up = {}
        self.pseudocost_down = {}
        self.pseudocost_up_count = {}
        self.pseudocost_down_count = {}
        
    def solve(self, objective: Dict, constraints: List[Dict], variables: List[Dict]) -> Dict:
        """求解 MILP 问题"""
        if not LP_SOLVER_AVAILABLE:
            return {
                'status': 'error',
                'message': '缺少底层 LP 求解器 (ultimate_lp_solver.py)',
                'is_feasible': False
            }
            
        print(f"\n{'='*60}")
        print(f"🌲 终极混合整数规划 (MILP) 分支定界求解开始 🌲")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        self.objective_type = objective['type'].lower()
        self.is_maximize = self.objective_type in ['max', 'maximize']
        
        # 初始化最优界
        self.incumbent_obj = -INF if self.is_maximize else INF
        self.incumbent_solution = None
        
        # 识别整数和二进制变量
        self.integer_vars = self._identify_integer_variables(variables)
        print(f"🔍 识别到 {len(self.integer_vars)} 个整数/二进制变量 (总变量数: {len(variables)})")
        
        if not self.integer_vars:
            print("ℹ️ 未检测到整数变量，直接作为纯连续线性规划 (LP) 求解。")
            lp_solver = UltimateLPSolver(**self.lp_kwargs)
            return lp_solver.solve(objective, constraints, variables)
            
        # 1. 根节点松弛求解
        root_node = MILPNode(bounds={})
        success = self._solve_node_relaxation(root_node, objective, constraints, variables)
        
        if not success or not root_node.is_feasible:
            print("❌ 根节点松弛问题无解，MILP 问题不可行。")
            return {
                'status': 'infeasible',
                'message': 'MILP 的连续松弛问题无解',
                'is_feasible': False,
                'solve_time': time.time() - self.start_time
            }
            
        print(f"🎯 根节点松弛最优界: {root_node.relaxation_obj:.6f}")
        
        # 1.5 尝试原始启发式算法 (Rounding Heuristic) 寻找初始解
        self._apply_rounding_heuristic(root_node.relaxation_solution, variables, objective, constraints)
        
        # 优先队列维护活动节点 (BBF: Best-Bound First)
        # 用堆来维护，堆是最小化。
        # 如果是最大化，存 -obj；如果是最小化，存 obj。
        active_nodes = []
        sort_key = -root_node.relaxation_obj if self.is_maximize else root_node.relaxation_obj
        heapq.heappush(active_nodes, (sort_key, self.nodes_explored, root_node))
        
        # 2. 开始分支定界循环
        while active_nodes:
            # 检查时间限制
            if time.time() - self.start_time > self.time_limit:
                print(f"⏳ 达到计算时间限制 ({self.time_limit}秒)，强制终止搜索。")
                break
                
            _, _, current_node = heapq.heappop(active_nodes)
            self.nodes_explored += 1
            
            # 定界 (Bound) / 剪枝 (Prune)
            # 如果当前节点的最好可能（松弛界）已经不如我们的现有最优整数解，剪枝
            if self._should_prune(current_node.relaxation_obj):
                # print(f"  ✂️ 剪枝: 目标值 {current_node.relaxation_obj:.4f} 不如现存最优解 {self.incumbent_obj:.4f}")
                self.nodes_pruned += 1
                continue
                
            # 检查节点解是否全部满足整数约束
            fractional_var, fractional_val = self._select_branching_variable(
                current_node.relaxation_solution, variables
            )
            
            if fractional_var is None:
                # 🎈 发现了一个整数可行解！
                self._update_incumbent(current_node.relaxation_obj, current_node.relaxation_solution)
                continue
                
            # 分支 (Branch)
            # x <= floor(v) 和 x >= ceil(v)
            left_bounds = copy.deepcopy(current_node.bounds)
            right_bounds = copy.deepcopy(current_node.bounds)
            
            # 获取原始变量的边界 (如果有)
            orig_lb, orig_ub = self._get_original_bounds(variables[fractional_var])
            
            # 只有当新分支的约束比原始约束和当前节点继承的约束更紧时才添加
            floor_val = math.floor(fractional_val)
            ceil_val = math.ceil(fractional_val)
            
            current_lb = current_node.bounds.get(fractional_var, (orig_lb, orig_ub))[0]
            current_ub = current_node.bounds.get(fractional_var, (orig_lb, orig_ub))[1]
            
            # 创建左节点 (x <= floor(v))
            if floor_val >= current_lb:
                # print(f"  🌿 创建左分支: {variables[fractional_var]['name']} <= {floor_val}")
                left_bounds[fractional_var] = (current_lb, min(current_ub, math.floor(fractional_val)))
                left_node = MILPNode(bounds=left_bounds, depth=current_node.depth + 1)
                
                # 求解左节点
                if self._solve_node_relaxation(left_node, objective, constraints, variables):
                    # 记录向下分支的伪代价
                    deterioration = current_node.relaxation_obj - left_node.relaxation_obj
                    if not self.is_maximize:
                        deterioration = left_node.relaxation_obj - current_node.relaxation_obj
                        
                    fraction_part = fractional_val - math.floor(fractional_val)
                    if fraction_part > 1e-6:
                        pseudo_c = deterioration / fraction_part
                        self.pseudocost_down[fractional_var] = self.pseudocost_down.get(fractional_var, 0.0) + pseudo_c
                        self.pseudocost_down_count[fractional_var] = self.pseudocost_down_count.get(fractional_var, 0) + 1

                    if not self._should_prune(left_node.relaxation_obj):
                        l_key = -left_node.relaxation_obj if self.is_maximize else left_node.relaxation_obj
                        heapq.heappush(active_nodes, (l_key, self.nodes_explored + 1, left_node))
                    else:
                        self.nodes_pruned += 1
                else:
                    self.nodes_pruned += 1
            else:
                 self.nodes_pruned += 1

            # 创建右节点 (x >= ceil(v))
            if ceil_val <= current_ub:
                # print(f"  🌿 创建右分支: {variables[fractional_var]['name']} >= {ceil_val}")
                right_bounds[fractional_var] = (max(current_lb, math.ceil(fractional_val)), current_ub)
                right_node = MILPNode(bounds=right_bounds, depth=current_node.depth + 1)
                
                # 求解右节点
                if self._solve_node_relaxation(right_node, objective, constraints, variables):
                    # 记录向上分支的伪代价
                    deterioration = current_node.relaxation_obj - right_node.relaxation_obj
                    if not self.is_maximize:
                        deterioration = right_node.relaxation_obj - current_node.relaxation_obj
                        
                    fraction_part = math.ceil(fractional_val) - fractional_val
                    if fraction_part > 1e-6:
                        pseudo_c = deterioration / fraction_part
                        self.pseudocost_up[fractional_var] = self.pseudocost_up.get(fractional_var, 0.0) + pseudo_c
                        self.pseudocost_up_count[fractional_var] = self.pseudocost_up_count.get(fractional_var, 0) + 1

                    if not self._should_prune(right_node.relaxation_obj):
                        r_key = -right_node.relaxation_obj if self.is_maximize else right_node.relaxation_obj
                        heapq.heappush(active_nodes, (r_key, self.nodes_explored + 2, right_node))
                    else:
                        self.nodes_pruned += 1
                else:
                    self.nodes_pruned += 1
            else:
                self.nodes_pruned += 1
                
            # 每隔一段时间汇报进度
            if self.nodes_explored % 50 == 0:
                gap_info = ""
                if self.incumbent_obj is not None and self.incumbent_obj not in [INF, -INF]:
                    best_possible = -active_nodes[0][0] if self.is_maximize and active_nodes else (active_nodes[0][0] if active_nodes else self.incumbent_obj)
                    if best_possible != 0:
                        gap = abs(best_possible - self.incumbent_obj) / abs(best_possible) * 100
                        gap_info = f" | Gap: {gap:.2f}%"
                        
                print(f"🌳 B&B进度: 探索了 {self.nodes_explored} 个节点 | "
                      f"剪枝: {self.nodes_pruned} | "
                      f"队列深度: {len(active_nodes)} | "
                      f"当前最优解: {'N/A' if self.incumbent_obj in [INF, -INF] else f'{self.incumbent_obj:.4f}'}"
                      f"{gap_info}")

        # 3. 结果汇总
        solve_time = time.time() - self.start_time
        print(f"\n✅ MILP 求解结束!")
        print(f"   耗时: {solve_time:.4f}秒")
        print(f"   探索节点总数: {self.nodes_explored}")
        print(f"   剪枝节点总数: {self.nodes_pruned}")
        
        if self.incumbent_solution is not None:
            print(f"   获得最优整数解: {self.incumbent_obj:.4f}")
            return {
                'status': 'optimal' if not active_nodes else 'feasible',
                'message': '寻找到整数最优解' if not active_nodes else '因超时返回当前最好整数解',
                'is_feasible': True,
                'objective_value': self.incumbent_obj,
                'solution': self.incumbent_solution,
                'solve_time': solve_time,
                'nodes_explored': self.nodes_explored
            }
        else:
            print(f"❌ 无法找到任何可行的整数解。")
            return {
                'status': 'infeasible',
                'message': 'MILP 问题无整数可行解',
                'is_feasible': False,
                'solve_time': solve_time,
                'nodes_explored': self.nodes_explored
            }
            
    def _solve_node_relaxation(self, node: MILPNode, objective: Dict, constraints: List[Dict], variables: List[Dict]) -> bool:
        """求解在具有特定额外边界条件的节点上的连续松弛问题"""
        
        # 1. 创建节点的变量副本（仅仅修改界限，用于此次 LP）
        node_variables = copy.deepcopy(variables)
        for idx, (lb, ub) in node.bounds.items():
            # 临时把离散变量当作连续变量处理
            vtype = node_variables[idx].get('type', 'nonneg').lower()
            if vtype in ['binary', 'integer']:
                 node_variables[idx]['type'] = 'nonneg'
            node_variables[idx] = copy.deepcopy(node_variables[idx])
            node_variables[idx]['bounds'] = [lb, ub]
            
        # 所有原本在基模型中的 integer/binary 现在也都降维成连续变量
        for idx in self.integer_vars:
            if idx not in node.bounds:
                vtype = node_variables[idx].get('type', 'nonneg').lower()
                if vtype in ['binary', 'integer']:
                     node_variables[idx]['type'] = 'nonneg'

        # 2. 调用底层极速 LP 求解器
        # 我们对于子节点，可以禁用冗长的打印
        lp_solver = UltimateLPSolver(**self.lp_kwargs)
        
        # 临时静音子求解器的标准输出，避免海量日志
        import os, sys, io
        old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
        try:
            result = lp_solver.solve(objective, constraints, node_variables)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        # 3. 解析结果
        if result['status'] == 'optimal' and result['solution'] is not None:
            node.is_feasible = True
            node.relaxation_solution = result['solution']
            node.relaxation_obj = result['objective_value']
            return True
        else:
            node.is_feasible = False
            return False

    def _should_prune(self, node_obj: float) -> bool:
        """判断是否应该剪枝此节点（Bound 约束）"""
        if self.incumbent_obj in [INF, -INF]:
            return False
            
        # 如果是最大化问题，松弛上限（节点的理论最好值）小于等于我们手头的现成最好解，不再探索
        # 注意加上容差，防止浮点误差导致的最优解被剪枝
        if self.is_maximize and node_obj <= self.incumbent_obj + self.tolerance:
            return True
            
        # 如果是最小化问题，松弛下限（节点的理论最好值）大于等于我们手头的现成最好解，不再探索
        if not self.is_maximize and node_obj >= self.incumbent_obj - self.tolerance:
            return True
            
        return False
        
    def _update_incumbent(self, obj: float, solution: np.ndarray):
        """更新现有的最优整数解 (Incumbent)"""
        is_better = False
        if self.incumbent_obj in [INF, -INF]:
            is_better = True
        elif self.is_maximize and obj > self.incumbent_obj + self.tolerance:
            is_better = True
        elif not self.is_maximize and obj < self.incumbent_obj - self.tolerance:
            is_better = True
            
        if is_better:
            self.incumbent_obj = obj
            self.incumbent_solution = copy.deepcopy(solution)
            print(f"  🌟 发现更好的整数可用解! 目标值: {obj:.6f}")

    def _get_original_bounds(self, var_dict: Dict) -> Tuple[float, float]:
        """获取变量的绝对原始边界"""
        bnd = var_dict.get('bounds', None)
        if bnd:
            lb = bnd[0] if bnd[0] is not None else -INF
            ub = bnd[1] if bnd[1] is not None else INF
            return lb, ub
            
        vtype = var_dict.get('type', 'nonneg').lower()
        if vtype == 'binary':
            return 0.0, 1.0
        elif vtype in ['free', 'unrestricted', 'unbounded']:
            return -INF, INF
        elif vtype in ['neg', 'negative']:
            return -INF, 0.0
        else: # 'nonneg', 'integer'
            return 0.0, INF
            
    def _identify_integer_variables(self, variables: List[Dict]) -> List[int]:
        """识别所有标记为 integer 或 binary 的变量索引"""
        indices = []
        for i, var in enumerate(variables):
            vtype = var.get('type', 'nonneg').lower()
            if vtype in ['integer', 'binary']:
                indices.append(i)
        return indices
        
    def _select_branching_variable(self, solution: np.ndarray, variables: List[Dict]) -> Tuple[Optional[int], Optional[float]]:
        """使用伪代价分数(Pseudo-cost)挑选最佳分支变量，替代传统的纯距离 0.5 最大化"""
        best_var = None
        best_score = -1.0
        best_val = None
        
        # 默认惩罚期望值 (当某个变量还没被探测过时的初始乐观估算值)
        DEFAULT_PCOST = 1e-4 
        
        for idx in self.integer_vars:
            val = float(solution[idx])
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)
            
            f_down = val - floor_val
            f_up = ceil_val - val
            
            # 如果违反了整数约束
            if min(f_down, f_up) > self.integer_tolerance:
                # 获取平均向下和向上伪代价估值
                pc_down_cnt = self.pseudocost_down_count.get(idx, 0)
                if pc_down_cnt > 0:
                    avg_pc_down = self.pseudocost_down[idx] / pc_down_cnt
                else:
                    avg_pc_down = DEFAULT_PCOST
                    
                pc_up_cnt = self.pseudocost_up_count.get(idx, 0)
                if pc_up_cnt > 0:
                    avg_pc_up = self.pseudocost_up[idx] / pc_up_cnt
                else:
                    avg_pc_up = DEFAULT_PCOST
                
                # 预估衰减期望乘积分支打分 = max(向下期望劣变, eps) * max(向上期望劣变, eps)
                score_down = max(avg_pc_down * f_down, 1e-6)
                score_up = max(avg_pc_up * f_up, 1e-6)
                score = score_down * score_up
                
                # 如果完全没历史数据，稍微补充一点传统的向心度打分以作区分兜底
                if pc_down_cnt == 0 and pc_up_cnt == 0:
                    score += 1e-6 * (0.5 - abs(0.5 - f_down))
                
                if score > best_score:
                    best_score = score
                    best_var = idx
                    best_val = val
                    
        return best_var, best_val

    def _apply_rounding_heuristic(self, relax_solution: np.ndarray, variables: List[Dict], objective: Dict, constraints: List[Dict]):
        """简单圆整启发式：尝试把松弛解就近圆整，如果依然满足所有约束，则得到一个免费的初始基准解。"""
        if relax_solution is None:
            return
            
        rounded_solution = copy.deepcopy(relax_solution)
        for idx in self.integer_vars:
            rounded_solution[idx] = round(relax_solution[idx])
            
        # 验证是否满足所有的边界和约束
        is_feasible_heuristic = True
        
        # 1. 验证变量自身范围
        for i, var in enumerate(variables):
            lb, ub = self._get_original_bounds(var)
            val = rounded_solution[i]
            if val < lb - self.tolerance or val > ub + self.tolerance:
                is_feasible_heuristic = False
                break
                
        # 2. 验证所有约束
        if is_feasible_heuristic:
            for c in constraints:
                lhs = sum(c['coeffs'][i] * rounded_solution[i] for i in range(len(c['coeffs'])))
                rhs = c['rhs']
                ctype = c['type']
                
                if ctype in ['<=', 'less_equal', 'le']:
                    if lhs > rhs + self.tolerance: is_feasible_heuristic = False
                elif ctype in ['>=', 'greater_equal', 'ge']:
                    if lhs < rhs - self.tolerance: is_feasible_heuristic = False
                elif ctype in ['=', 'eq', 'equal']:
                    if abs(lhs - rhs) > self.tolerance: is_feasible_heuristic = False
                    
                if not is_feasible_heuristic:
                    break
                    
        # 如果启发式找出来的解竟然是可行的！算一下目标值
        if is_feasible_heuristic:
            obj_val = sum(objective['coeffs'][i] * rounded_solution[i] for i in range(len(objective['coeffs'])))
            print(f"  ⛏️ 原始启发式算法 (Rounding Heuristic) 成功找到并初始化整数可行解!")
            self._update_incumbent(obj_val, rounded_solution)

# === 独立测试入口 ===
if __name__ == "__main__":
    print("运行 MILP (分支定界) 测试...")
    
    # 一个简单的整数背包问题/资源分配问题
    # Max Z = 5 x1 + 8 x2
    # s.t. x1 + x2 <= 6
    #      5 x1 + 9 x2 <= 45
    #      x1, x2 为整数 (>0)
    
    test_obj = {'type': 'maximize', 'coeffs': [5, 8]}
    test_constraints = [
        {'type': '<=', 'coeffs': [1, 1], 'rhs': 6},
        {'type': '<=', 'coeffs': [5, 9], 'rhs': 45}
    ]
    test_vars = [
        {'name': 'x1', 'type': 'integer'},
        {'name': 'x2', 'type': 'integer'}
    ]
    
    solver = MILPBranchAndBound(solver='builtin') # 可以改为 'pdhg' 测试异构硬件加速子节点
    result = solver.solve(test_obj, test_constraints, test_vars)
    
    if result.get('solution') is not None:
         print(f"测试通过! 解: {result['solution']}, 目标: {result['objective_value']}")
