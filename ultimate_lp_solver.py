#!/usr/bin/env python3
"""
终极线性规划求解器核心代码
真正处理所有复杂情况：
- 任意多个变量
- 复杂的min/max函数
- 混乱的不等式约束
- 负数参数
- 无最优解检测
"""

import numpy as np
import scipy.sparse as sp
import time
import sys
import io
from typing import List, Dict, Tuple, Optional, Union

# 设置标准输出为 UTF-8 以避免 Windows 下的 gbk 编码错误
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 引入 PDHG 求解器核心
try:
    from pdhg_accelerated_solver import PDHGSolver
    PDHG_AVAILABLE = True
    print("[OK] PDHG (Chambolle-Pock) 一阶算法模块已加载")
except ImportError:
    PDHG_AVAILABLE = False
    print("[WARN] 未找到 pdhg_accelerated_solver.py")

# 引入高级预处理器
try:
    from presolver import AdvancedPresolver
    PRESOLVE_AVAILABLE = True
except ImportError:
    PRESOLVE_AVAILABLE = False

# 引入稀疏修正单纯形法求解器
try:
    from sparse_revised_simplex import SparseRevisedSimplex
    SPARSE_SIMPLEX_AVAILABLE = True
    print("[OK] SparseRevisedSimplex 稀疏基元求解模块已加载")
except ImportError:
    SPARSE_SIMPLEX_AVAILABLE = False
    print("[WARN] 未找到 sparse_revised_simplex.py")

# 导入CuPy用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy已导入，支持GPU加速")
except ImportError:
    print("⚠️ CuPy未安装，将使用CPU计算")
    CUPY_AVAILABLE = False

# 导入PuLP求解器
try:
    from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, LpStatusOptimal, LpStatusInfeasible, LpStatusUnbounded
    PULP_AVAILABLE = True
    print("✅ PuLP求解器已导入")
except ImportError:
    print("⚠️ PuLP求解器未安装，将使用内置求解器")
    PULP_AVAILABLE = False

class UltimateLPSolver:
    """
    终极线性规划求解器
    处理任意复杂的线性规划问题
    """
    
    def __init__(self, tolerance=1e-8, solver='auto', use_gpu=False, use_npu=False, npu_cores=2):
        self.tolerance = tolerance
        self.use_npu = use_npu
        self.npu_cores = npu_cores
        
        # 选择求解器
        if solver == 'auto':
            if SPARSE_SIMPLEX_AVAILABLE:
                self.solver = 'revised_simplex'
            elif PDHG_AVAILABLE:
                self.solver = 'pdhg'
            else:
                self.solver = 'pulp' if PULP_AVAILABLE else 'builtin'
        elif solver in ['pulp', 'builtin', 'pdhg', 'revised_simplex']:
            if solver == 'pulp' and not PULP_AVAILABLE:
                print(f"⚠️ PuLP求解器不可用，将使用内置求解器")
                self.solver = 'builtin'
            elif solver == 'pdhg' and not PDHG_AVAILABLE:
                print(f"⚠️ PDHG求解器不可用，降级使用内置求解器")
                self.solver = 'builtin'
            elif solver == 'revised_simplex' and not SPARSE_SIMPLEX_AVAILABLE:
                self.solver = 'builtin'
            else:
                self.solver = solver
        else:
            print(f"⚠️ 未知求解器: {solver}，将使用内置求解器")
            self.solver = 'builtin'
        
        # 设置GPU加速
        if use_gpu and CUPY_AVAILABLE:
            self.use_gpu = True
            self.array_lib = cp  # 使用CuPy进行GPU计算
        else:
            self.use_gpu = False
            self.array_lib = np  # 使用NumPy进行CPU计算
        
        print(f"🎯 终极线性规划求解器已启动")
        print(f"   - 容差: {tolerance}")
        print(f"   - 求解器: {'PuLP' if self.solver == 'pulp' else '内置求解器'}")
        print(f"   - 计算设备: {'GPU' if self.use_gpu else 'NPU' if self.use_npu else 'CPU'}")
    
    def solve(self, objective: Dict, constraints: List[Dict], variables: List[Dict]) -> Dict:
        """
        求解复杂的线性规划问题
        
        参数:
            objective: {'type': 'max'/'min', 'coeffs': [c1, c2, ...]}
            constraints: [
                {'type': '<='/'≥'/'=', 'coeffs': [a1, a2, ...], 'rhs': b, 'name': 'constraint_name'}
            ]
            variables: [
                {'name': 'x1', 'type': 'free'/'nonneg'/'pos'/'neg', 'bounds': [low, high]}
            ]
        
        返回:
            详细结果字典
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🔥 终极线性规划求解开始 🔥")
        print(f"{'='*60}")
        
        # 解析问题
        print(f"\n📋 问题解析:")
        print(f"   目标函数: {objective['type'].upper()}")
        obj_str = " + ".join([f"{coeff:.3f}*{var['name']}" for var, coeff in zip(variables, objective['coeffs'])])
        print(f"   {obj_str}")
        print(f"   变量数量: {len(variables)}")
        print(f"   约束数量: {len(constraints)}")
        
        try:
            # 根据选择的求解器执行不同的求解逻辑
            if self.solver == 'pulp':
                print(f"\n🔄 使用PuLP求解器进行求解...")
                final_result = self._pulp_solve(objective, constraints, variables)
            elif self.solver == 'pdhg':
                print(f"\n🔄 使用PDHG (NPU 友好一阶算法) 进行极速求解...")
                final_result = self._pdhg_solve(objective, constraints, variables)
            elif self.solver == 'revised_simplex':
                print(f"\n🔄 使用 Revised Simplex (稀疏 LU 分解) 进行极速求解...")
                standard_form = self._standardize_problem(objective, constraints, variables)
                if PRESOLVE_AVAILABLE:
                    standard_form = self._presolve(standard_form)
                final_result = self._sparse_revised_simplex_solve(standard_form['objective'], standard_form['constraints'], standard_form['variables'])
                # Postsolve
                if PRESOLVE_AVAILABLE and hasattr(self, '_active_presolver') and final_result['status'] == 'optimal':
                    presolved_vars = standard_form['variables']
                    full_solution = self._active_presolver.postsolve(final_result['solution'].tolist(), presolved_vars)
                    final_result['solution'] = np.array(full_solution)
                    if 'offset' in standard_form['objective']:
                        final_result['objective_value'] += standard_form['objective']['offset']
            else:
                # 1. 问题标准化
                print(f"\n🔄 第1步: 问题标准化...")
                standard_form = self._standardize_problem(objective, constraints, variables)
                
                # 1.5. 预处理 (Presolve)
                print(f"🔄 第1.5步: 问题预处理 (Presolving)...")
                standard_form = self._presolve(standard_form)

                # 2. 构建单纯形表
                print(f"🔄 第2步: 构建单纯形表...")
                tableau_info = self._build_tableau(standard_form)
                
                # 3. 求解
                print(f"🔄 第3步: 使用内置求解器进行求解...")
                result = self._solve_problem(tableau_info, standard_form)
            
                # 4. 结果后处理
                result = self._process_result(result, standard_form, variables)
                
                # 4.5 预处理恢复 (Postsolve)
                if PRESOLVE_AVAILABLE and hasattr(self, '_active_presolver'):
                    print(f"🔄 第4.5步: 恢复预处理变量 (Postsolve)...")
                    if result['status'] == 'optimal' and result['solution'] is not None:
                        presolved_vars = standard_form['variables']
                        full_solution = self._active_presolver.postsolve(result['solution'].tolist(), presolved_vars)
                        result['solution'] = np.array(full_solution)
                        
                        # Fix objective value if offset exists
                        if 'offset' in standard_form['objective']:
                            result['objective_value'] += standard_form['objective']['offset']
                
                final_result = result
            
            final_result['solve_time'] = time.time() - start_time
            final_result['is_feasible'] = final_result['status'] == 'optimal'
            
            # 5. 详细验证
            if final_result['status'] == 'optimal':
                self._validate_solution(final_result, objective, constraints, variables)
            
            print(f"\n✅ 求解完成!")
            print(f"   状态: {final_result['status']}")
            print(f"   耗时: {final_result['solve_time']:.4f}秒")
            print(f"   是否可行: {'是' if final_result['is_feasible'] else '否'}")
            
            return final_result
        except Exception as e:
            print(f"\n❌ 求解失败: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'solve_time': time.time() - start_time,
                'is_feasible': False,
                'solution': None,
                'objective_value': None
            }
    
    def _standardize_problem(self, objective, constraints, variables):
        """标准化问题"""
        print(f"   📐 标准化约束...")
        
        # 处理约束
        processed_constraints = []
        for i, constraint in enumerate(constraints):
            constraint = constraint.copy()
            
            # 标准化约束类型
            ctype = constraint['type'].lower()
            if ctype in ['<=', 'less_equal', 'le']:
                ctype = '<='
            elif ctype in ['>=', 'greater_equal', 'ge']:
                ctype = '>='
            elif ctype in ['=', 'eq', 'equal']:
                ctype = '='
            else:
                raise ValueError(f"不支持的约束类型: {constraint['type']}")
            
            rhs = constraint['rhs']
            coeffs = constraint['coeffs']
            
            if rhs < -self.tolerance:
                rhs = -rhs
                coeffs = [-c for c in coeffs]
                if ctype == '<=':
                    ctype = '>='
                elif ctype == '>=':
                    ctype = '<='
            
            constraint['type'] = ctype
            constraint['rhs'] = rhs
            constraint['coeffs'] = coeffs
            
            processed_constraints.append(constraint)
        
        # 处理变量和边界约束
        variable_types = []
        n_vars = len(variables)
        for i, var in enumerate(variables):
            var_type = var.get('type', 'nonneg').lower()
            if var_type in ['free', 'unrestricted', 'unbounded']:
                variable_types.append('free')
            elif var_type in ['nonneg', 'positive', 'pos', 'binary', 'integer']:  # 将binary和integer变量当作nonneg处理
                variable_types.append('nonneg')
            elif var_type in ['neg', 'negative']:
                variable_types.append('neg')
            else:
                variable_types.append('nonneg')
                
            # 处理额外的边界 bounds: [lb, ub]
            bounds = var.get('bounds', None)
            if bounds is not None:
                lb, ub = bounds[0], bounds[1]
                if lb is not None:
                    # 对于 nonneg 变量，内置约束为 x_i >= 0
                    if var_type == 'free' or lb > self.tolerance:
                        coeffs = [0.0] * n_vars
                        coeffs[i] = 1.0
                        c_type = '>='
                        c_rhs = lb
                        if c_rhs < -self.tolerance:
                            c_rhs = -c_rhs
                            coeffs = [-c for c in coeffs]
                            c_type = '<='
                        processed_constraints.append({
                            'type': c_type,
                            'coeffs': coeffs,
                            'rhs': c_rhs,
                            'name': f"{var.get('name', f'x{i}')}_lb"
                        })
                # x_i <= ub => 添加约束 x_i <= ub
                if ub is not None and ub < float('inf'):
                    coeffs = [0.0] * n_vars
                    coeffs[i] = 1.0
                    c_type = '<='
                    c_rhs = ub
                    if c_rhs < -self.tolerance:
                        c_rhs = -c_rhs
                        coeffs = [-c for c in coeffs]
                        c_type = '>='
                    processed_constraints.append({
                        'type': c_type,
                        'coeffs': coeffs,
                        'rhs': c_rhs,
                        'name': f"{var.get('name', f'x{i}')}_ub"
                    })
        
        # 收集原始变量类型用于后续验证
        original_variable_types = [var.get('type', 'nonneg').lower() for var in variables]
        
        print(f"   ✅ 标准化完成")
        print(f"      变量类型: [显示前10个] {variable_types[:10]} ...")
        print(f"      约束类型: [显示前10个] {[c['type'] for c in processed_constraints][:10]} ...")
        
        return {
            'objective': objective,
            'constraints': processed_constraints,
            'variables': variables,
            'variable_types': variable_types,
            'original_variable_types': original_variable_types,  # 添加原始变量类型
            'n_original_vars': len(variables)
        }
    
    def _presolve(self, problem):
        """预处理模块 (Presolve): 剔除冗余约束和变量"""
        if PRESOLVE_AVAILABLE:
            self._active_presolver = AdvancedPresolver(tolerance=self.tolerance)
            return self._active_presolver.presolve(problem)
            
        print(f"   🧹 运行粗颗粒预处理 (由于找不到 presolver.py，仅使用简单逻辑)...")
        constraints = problem['constraints']
        orig_count = len(constraints)
        
        valid_constraints = []
        for c in constraints:
            coeffs = np.array(c['coeffs'])
            # 如果所有系数都极其趋近 0
            if np.all(np.abs(coeffs) < self.tolerance):
                # 检查 RHS 是否矛盾
                if c['type'] == '<=' and 0 <= c['rhs']:
                    pass # 冗余 0 <= b
                elif c['type'] == '>=' and 0 >= c['rhs']:
                    pass # 冗余 0 >= b
                elif c['type'] == '=' and abs(0 - c['rhs']) < self.tolerance:
                    pass # 冗余 0 == 0
                else:
                    raise ValueError(f"检测到绝对矛盾空约束: 0 {c['type']} {c['rhs']}")
            else:
                valid_constraints.append(c)

        eliminated = orig_count - len(valid_constraints)
        if eliminated > 0:
            print(f"   🧹 预处理移除了 {eliminated} 个冗余的零截距约束。保留 {len(valid_constraints)} 个约束矩阵行。")
            problem['constraints'] = valid_constraints

        return problem
    
    def _build_tableau(self, problem):
        """构建单纯形表"""
        print(f"   📊 构建表格...")
        
        constraints = problem['constraints']
        variable_types = problem['variable_types']
        n_vars = len(constraints[0]['coeffs']) if constraints else 0
        
        # 统计需要的变量 - 重要：>=约束也需要人工变量
        n_slack = sum(1 for c in constraints if c['type'] == '<=')
        n_artificial = sum(1 for c in constraints if c['type'] == '=') + sum(1 for c in constraints if c['type'] == '>=')
        n_surplus = sum(1 for c in constraints if c['type'] == '>=')
        
        # 处理自由变量（拆分）
        expanded_coeffs, variable_mapping, processed_constraints = self._handle_free_variables(problem)
        n_expanded_vars = len(expanded_coeffs)
        n_processed_constraints = len(processed_constraints)
        
        # 构建表格
        n_total_vars = n_expanded_vars + n_slack + n_surplus + n_artificial
        tableau = self.array_lib.zeros((n_processed_constraints + 1, n_total_vars + 1), dtype=float)
        
        # 填充约束
        slack_idx = 0
        surplus_idx = 0
        artificial_idx = 0
        
        for i, constraint in enumerate(processed_constraints):
            # 填充系数
            tableau[i, :n_expanded_vars] = constraint['coeffs']
            
            # 添加松弛变量、冗余变量或人工变量
            if constraint['type'] == '<=':
                tableau[i, n_expanded_vars + slack_idx] = 1.0
                slack_idx += 1
            elif constraint['type'] == '>=':  # >= 约束：-冗余变量 + 人工变量
                tableau[i, n_expanded_vars + n_slack + surplus_idx] = -1.0
                tableau[i, n_expanded_vars + n_slack + n_surplus + artificial_idx] = 1.0
                surplus_idx += 1
                artificial_idx += 1
            elif constraint['type'] == '=':
                tableau[i, n_expanded_vars + n_slack + n_surplus + artificial_idx] = 1.0
                artificial_idx += 1
            
            # 填充右端值
            tableau[i, -1] = constraint['rhs']
        
        # 填充目标函数行 - 标准化为最小化问题
        # 对于最小化：Tableau[-1, :] = -coefficients
        # 对于最大化：Tableau[-1, :] = coefficients （在single_phase中会取负）
        tableau[-1, :n_expanded_vars] = self.array_lib.array(expanded_coeffs)
        
        # 设置初始基 - 对于<=约束，松弛变量是基变量；对于>=约束，初始基不可行
        # 需要处理等式约束的情况
        
        print(f"   🎯 初始基设置:")
        basic_vars = []
        for i in range(n_expanded_vars, n_expanded_vars + n_slack):
            basic_vars.append(f"松弛变量{i-n_expanded_vars+1}")
        for i in range(n_expanded_vars + n_slack, n_expanded_vars + n_slack + n_artificial):
            basic_vars.append(f"人工变量{i-(n_expanded_vars+n_slack)+1}")
        
        print(f"      初始基变量: {basic_vars}")
        print(f"      初始基解: 所有原始变量=0，基变量=约束右端值")
        
        print(f"   ✅ 表格构建完成")
        print(f"      表格大小: {tableau.shape}")
        print(f"      原始变量: {n_vars} → 展开变量: {n_expanded_vars}")
        print(f"      约束: {n_processed_constraints}, 松弛变量: {n_slack}, 冗余变量: {n_surplus}, 人工变量: {n_artificial}")
        
        return {
            'tableau': tableau,
            'n_constraints': n_processed_constraints,
            'n_expanded_vars': n_expanded_vars,
            'n_slack': n_slack,
            'n_surplus': n_surplus,
            'n_artificial': n_artificial,
            'variable_mapping': variable_mapping,
            'objective_coeffs': expanded_coeffs,
            'processed_constraints': processed_constraints
        }
    
    def _handle_free_variables(self, problem):
        """处理自由变量"""
        objective = problem['objective']
        constraints = problem['constraints']
        variable_types = problem['variable_types']
        n_vars = len(objective['coeffs'])
        
        # 处理目标函数系数
        expanded_coeffs = []
        variable_mapping = []
        
        for i, var_type in enumerate(variable_types):
            coeff = objective['coeffs'][i]
            
            if var_type == 'free':
                # 自由变量: xi = xi+ - xi-
                expanded_coeffs.extend([coeff, -coeff])
                variable_mapping.append({
                    'original_index': i,
                    'positive_var': len(expanded_coeffs) - 2,
                    'negative_var': len(expanded_coeffs) - 1
                })
            else:
                expanded_coeffs.append(coeff)
                variable_mapping.append({
                    'original_index': i,
                    'positive_var': len(expanded_coeffs) - 1,
                    'negative_var': None
                })
        
        # 处理约束系数
        processed_constraints = []
        for constraint in constraints:
            coeffs = self.array_lib.array(constraint['coeffs'], dtype=float)
            processed_coeffs = []
            
            for i, var_type in enumerate(variable_types):
                coeff = coeffs[i]
                
                if var_type == 'free':
                    # 自由变量: xi = xi+ - xi-
                    processed_coeffs.extend([coeff, -coeff])
                else:
                    processed_coeffs.append(coeff)
            
            processed_constraint = constraint.copy()
            processed_constraint['coeffs'] = self.array_lib.array(processed_coeffs, dtype=float)
            processed_constraints.append(processed_constraint)
        
        return expanded_coeffs, variable_mapping, processed_constraints
    
    def _solve_problem(self, tableau_info, problem):
        """求解问题"""
        tableau = tableau_info['tableau'].copy()
        n_constraints = tableau_info['n_constraints']
        n_artificial = tableau_info['n_artificial']
        objective = problem['objective']
        
        # 两阶段法
        if n_artificial > 0:
            result = self._two_phase_simplex(tableau, n_constraints, tableau_info, objective)
        else:
            result = self._single_phase_simplex(tableau, n_constraints, objective)
        
        result['tableau_info'] = tableau_info
        return result
    
    def _two_phase_simplex(self, tableau, n_constraints, tableau_info, objective):
        """两阶段单纯形法"""
        print(f"      🔄 使用两阶段法...")
        
        # 阶段1: 消除人工变量
        phase1_result = self._eliminate_artificial_variables(tableau, n_constraints, tableau_info)
        
        if phase1_result['status'] != 'optimal':
            return {
                'status': 'infeasible',
                'message': '问题无可行解',
                'tableau': phase1_result.get('tableau')
            }
        
        # 移除人工变量和冗余变量
        tableau = phase1_result['tableau']
        
        # 先移除冗余变量
        surplus_start = tableau_info['n_expanded_vars'] + tableau_info['n_slack']
        if tableau_info['n_surplus'] > 0:
            tableau = np.delete(tableau, range(surplus_start, surplus_start + tableau_info['n_surplus']), axis=1)
            # 移除冗余变量后，人工变量的起始位置变为 n_expanded_vars + n_slack
            new_artificial_start = tableau_info['n_expanded_vars'] + tableau_info['n_slack']
        else:
            # 如果没有移除冗余变量，人工变量起始位置不变
            new_artificial_start = tableau_info['n_expanded_vars'] + tableau_info['n_slack'] + tableau_info['n_surplus']
        
        # 然后移除人工变量
        if tableau_info['n_artificial'] > 0:
            # 确保人工变量起始位置在有效范围内
            if new_artificial_start < tableau.shape[1] and new_artificial_start + tableau_info['n_artificial'] <= tableau.shape[1]:
                tableau = np.delete(tableau, range(new_artificial_start, new_artificial_start + tableau_info['n_artificial']), axis=1)
            else:
                # 如果人工变量已经在迭代中被替换，可以跳过移除步骤
                print(f"      ℹ️  人工变量可能已被替换，跳过移除步骤")
        
        # 阶段2: 恢复原目标函数并求解原问题
        print(f"      🔄 阶段2：恢复原目标函数并求解原问题")
        
        # 恢复原目标函数（仅保留有效变量的系数）
        original_obj = self.array_lib.zeros(tableau.shape[1])
        original_coeffs = tableau_info['objective_coeffs']
        
        # 确保只使用原目标函数中有效变量的系数
        n_valid_vars = min(len(original_coeffs), tableau.shape[1] - 1)  # 最后一列是RHS
        original_obj[:n_valid_vars] = self.array_lib.array(original_coeffs[:n_valid_vars])
        
        tableau[-1, :] = original_obj
        print(f"      📊 原目标函数恢复完成: {original_obj[:n_valid_vars]}")
        
        phase2_result = self._single_phase_simplex(tableau, n_constraints, objective)
        
        return {
            'status': phase2_result['status'],
            'message': phase2_result.get('message', ''),
            'tableau': phase2_result.get('tableau'),
            'iterations': phase2_result.get('iterations', 0)
        }
    
    def _eliminate_artificial_variables(self, tableau, n_constraints, tableau_info):
        """消除人工变量"""
        print(f"      🔄 开始阶段1：消除人工变量")
        
        # 构建辅助目标函数
        aux_obj = self.array_lib.zeros(tableau.shape[1])
        artificial_start = tableau_info['n_expanded_vars'] + tableau_info['n_slack'] + tableau_info['n_surplus']
        
        print(f"      📊 人工变量起始位置: {artificial_start}")
        print(f"      📊 人工变量数量: {tableau_info['n_artificial']}")
        
        for i in range(tableau_info['n_artificial']):
            aux_obj[artificial_start + i] = 1.0
        
        # 更新目标函数行
        tableau[-1, :] = aux_obj
        
        # 将人工变量出基（修正目标函数行使其 reduced cost = 0）
        for i, constraint in enumerate(tableau_info.get('processed_constraints', [])):
            if constraint['type'] in ['>=', '=']:
                tableau[-1, :] -= tableau[i, :]
        
        print(f"      📊 辅助目标函数(修正后): {tableau[-1, :]}")
        print(f"      🔧 开始阶段1优化...")
        
        # 迭代优化
        result = self._simplex_iterations(tableau, n_constraints, max_iter=1000)
        
        print(f"      📊 阶段1结果: {result['status']}")
        return result
    
    def _single_phase_simplex(self, tableau, n_constraints, objective):
        """单阶段单纯形法"""
        # 如果是最大化，转换为最小化问题
        if objective['type'].lower() in ['max', 'maximize']:
            print(f"      🔄 转换最大化问题为最小化问题")
            tableau[-1, :] = -tableau[-1, :]
            print(f"      📊 目标函数行转换: 取负号")
        
        return self._simplex_iterations(tableau, n_constraints, max_iter=1000)
    
    def _simplex_iterations(self, tableau, n_constraints, max_iter=1000):
        """单纯形法迭代"""
        iteration = 0
        
        print(f"      🔄 开始单纯形法迭代...")
        
        while iteration < max_iter:
            # 检查最优性 - 对于最小化问题，检验数应该 >= 0
            last_row = tableau[-1, :-1]
            
            # 寻找最小检验数（对于最小化，最小检验数应该 <= 0）
            min_reduced_cost = self.array_lib.min(last_row)
            
            print(f"      📊 迭代{iteration}: 最小检验数 = {min_reduced_cost:.6f}")
            
            if min_reduced_cost >= -self.tolerance:
                print(f"      ✅ 达到最优解，迭代次数: {iteration}")
                return {
                    'status': 'optimal',
                    'tableau': tableau,
                    'iterations': iteration
                }
            
            # 选择枢轴列（最小检验数列）
            pivot_col = self.array_lib.argmin(last_row)
            
            print(f"      🎯 选择枢轴列: {pivot_col}")
            
            # 寻找枢轴行 - 使用最小比值测试
            pivot_col_vals = tableau[:n_constraints, pivot_col]
            rhs_vals = tableau[:n_constraints, -1]
            
            print(f"      📈 枢轴列值: {pivot_col_vals}")
            print(f"      📊 RHS值: {rhs_vals}")
            
            # 检查无界性 - 所有值都 <= tolerance
            positive_mask = pivot_col_vals > self.tolerance
            if not self.array_lib.any(positive_mask):
                print(f"      ⚠️ 检测到无界问题")
                return {
                    'status': 'unbounded',
                    'message': '问题无界'
                }
            
            # 最小比值测试
            ratios = self.array_lib.full(n_constraints, self.array_lib.inf)
            ratios[positive_mask] = rhs_vals[positive_mask] / pivot_col_vals[positive_mask]
            
            print(f"      📐 比值: {ratios}")
            
            # 寻找最小正比值
            min_ratio = self.array_lib.min(ratios[positive_mask])
            if not self.array_lib.isfinite(min_ratio):
                print(f"      ⚠️ 检测到无界问题")
                return {
                    'status': 'unbounded',
                    'message': '问题无界'
                }
            
            # 找到最小比值对应的行
            min_ratio_idx = self.array_lib.argmin(ratios)
            pivot_row = min_ratio_idx
            
            print(f"      🎯 选择枢轴行: {pivot_row}, 比值: {min_ratio:.6f}")
            
            # 检查枢轴元素
            pivot_element = tableau[pivot_row, pivot_col]
            if abs(pivot_element) < self.tolerance:
                print(f"      ❌ 枢轴元素过小: {pivot_element}")
                return {
                    'status': 'numerical_error',
                    'message': '数值计算错误：枢轴元素过小'
                }
            
            # 执行枢轴运算
            print(f"      🔧 执行枢轴运算...")
            tableau = self._pivot(tableau, pivot_row, pivot_col)
            iteration += 1
            
            # 打印当前基解
            current_obj = tableau[-1, -1]
            print(f"      📊 当前目标值: {current_obj:.6f}")
        
        print(f"      ⚠️ 达到最大迭代次数: {max_iter}")
        return {
            'status': 'iterations_exceeded',
            'message': '达到最大迭代次数'
        }
    
    def _pivot(self, tableau, pivot_row, pivot_col):
        """枢轴运算"""
        pivot_element = tableau[pivot_row, pivot_col]
        
        # 枢轴行归一化
        tableau[pivot_row, :] /= pivot_element
        
        # 其他行消元
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
        
        return tableau
    
    def _process_result(self, result, problem, variables):
        """处理结果"""
        if result['status'] != 'optimal':
            return {
                'status': result['status'],
                'message': result.get('message', ''),
                'solution': None,
                'objective_value': None,
                'is_feasible': False
            }
        
        # 提取解
        tableau = result['tableau']
        tableau_info = result['tableau_info']
        
        solution = self._extract_solution(tableau, tableau_info, problem)
        objective_value = tableau[-1, -1]
        
        return {
            'status': 'optimal',
            'solution': solution,
            'objective_value': objective_value,
            'message': result.get('message', ''),
            'is_feasible': True,
            'tableau': tableau,
            'iterations': result.get('iterations', 0)
        }
    
    def _extract_solution(self, tableau, tableau_info, problem):
        """提取解"""
        n_vars = problem['n_original_vars']
        solution = self.array_lib.zeros(n_vars)
        variable_mapping = tableau_info['variable_mapping']
        
        # print(f"      🔍 解提取 - 最终单纯形表:")
        # print(f"      {tableau}")
        
        # 分析最终单纯形表，识别基变量
        n_constraints = tableau_info['n_constraints']
        
        # 检查原始变量
        for var_info in variable_mapping:
            orig_idx = var_info['original_index']
            pos_var = var_info['positive_var']
            neg_var = var_info['negative_var']
            
            # 检查正变量
            if pos_var < tableau.shape[1] - 1:
                col = tableau[:n_constraints, pos_var]
                # print(f"      📊 变量{orig_idx}: 列{pos_var} = {col}")
                
                # 检查是否为基变量：列中恰好有一个1.0，其余为0
                if self.array_lib.sum(self.array_lib.abs(col)) == 1.0:
                    row = self.array_lib.argmax(self.array_lib.abs(col))
                    if abs(col[row]) > 0.5:  # 确保是1.0而不是其他值
                        solution[orig_idx] = tableau[row, -1]
                        print(f"      ✅ 基变量: 变量{orig_idx} = {solution[orig_idx]} (行{row})")
                    else:
                        pass # print(f"      ⚠️ 非基变量: 变量{orig_idx} = 0")
                else:
                    pass # print(f"      ⚠️ 非基变量: 变量{orig_idx} = 0")
            
            # 如果是自由变量，减去负变量
            if neg_var is not None and neg_var < tableau.shape[1] - 1:
                col = tableau[:n_constraints, neg_var]
                # print(f"      📊 自由变量负部: 列{neg_var} = {col}")
                
                if self.array_lib.sum(self.array_lib.abs(col)) == 1.0:
                    row = self.array_lib.argmax(self.array_lib.abs(col))
                    if abs(col[row]) > 0.5:
                        neg_value = tableau[row, -1]
                        solution[orig_idx] -= neg_value
                        print(f"      📉 减去负部: -{neg_value} (行{row})")
        
        # 如果使用CuPy，将结果转换为NumPy数组
        if self.use_gpu:
            solution = cp.asnumpy(solution)
        
        print(f"      📊 最终解: {solution}")
        return solution
    
    def _pdhg_solve(self, objective, constraints, variables):
        """调用超快速的基于稀疏矩阵操作的 PDHG 核心引擎"""
        try:
            # 构建稀疏矩阵 A 和 边界
            m = len(constraints)
            n = len(variables)
            
            c = np.array(objective['coeffs'], dtype=np.float64)
            if objective['type'].lower() in ['max', 'maximize']:
                c = -c  # pdhg_solver 默认求最小 min c^T x
                
            data = []
            row_idx = []
            col_idx = []
            b = np.zeros(m)
            constraint_types = np.zeros(m) # 1 为 =, 0 为 <=
            
            for r, c_dict in enumerate(constraints):
                coeffs = c_dict['coeffs']
                for c_id, val in enumerate(coeffs):
                    if abs(val) > 1e-12:
                        row_idx.append(r)
                        col_idx.append(c_id)
                        data.append(float(val))
                        
                rhs = c_dict['rhs']
                ctype = c_dict['type']
                
                # 如果是 >=，标准化为 <=
                if ctype == '>=':
                    b[r] = -rhs
                    for i in range(len(row_idx)-1, -1, -1):
                        if row_idx[i] == r:
                            data[i] = -data[i]
                    constraint_types[r] = 0
                else:
                    b[r] = rhs
                    if ctype == '=':
                        constraint_types[r] = 1
                    else: # '<='
                        constraint_types[r] = 0

            # 稀疏约束矩阵
            A_sparse = sp.csc_matrix((data, (row_idx, col_idx)), shape=(m, n))
            
            # 构建解空间边界
            bounds = np.zeros((n, 2))
            for i, var in enumerate(variables):
                var_type = var.get('type', 'nonneg').lower()
                bnd = var.get('bounds', None)
                if bnd:
                    bounds[i, 0] = bnd[0]
                    bounds[i, 1] = bnd[1]
                else:
                    if var_type in ['free', 'unrestricted', 'unbounded']:
                        bounds[i, 0] = -np.inf
                        bounds[i, 1] = np.inf
                    elif var_type in ['neg', 'negative']:
                        bounds[i, 0] = -np.inf
                        bounds[i, 1] = 0
                    else: # 'nonneg' 等
                        bounds[i, 0] = 0
                        bounds[i, 1] = np.inf

            # 调用引擎
            pdhg_engine = PDHGSolver(tolerance=self.tolerance, use_npu=self.use_npu, npu_cores=self.npu_cores)
            result = pdhg_engine.solve(c, A_sparse, b, bounds, constraint_types)

            # 复原目标值 (如果是max的话，目标函数是负的) 
            if objective['type'].lower() in ['max', 'maximize']:
                result['objective_value'] = -result['objective_value']
            
            # 处理消息与一致性
            is_feasible = (result['status'] == 'optimal')
            result['is_feasible'] = is_feasible
            if is_feasible:
                result['message'] = 'PDHG 超级加速计算成功收敛'
            else:
                result['message'] = '极大迭代次数内未能收敛，或检测到不稳定的原对偶残差。'
                
            return result
        except Exception as e:
            print(f"❌ PDHG 求解失败: {str(e)}")
            raise e
    
    def _sparse_revised_simplex_solve(self, objective, constraints, variables):
        """调用稀疏修正单纯形法（基于 LU 分解）引擎"""
        try:
            expanded_vars_info = []
            c_expanded = []
            
            # 第1步：处理变量（包括自由变量转化）
            var_idx = 0
            for i, var in enumerate(variables):
                vtype = var.get('type', 'nonneg').lower()
                orig_c = objective['coeffs'][i]
                
                if vtype in ['free', 'unrestricted', 'unbounded']:
                    expanded_vars_info.append({'orig_idx': i, 'type': 'free', 'pos_idx': var_idx, 'neg_idx': var_idx + 1})
                    c_expanded.extend([orig_c, -orig_c])
                    var_idx += 2
                elif vtype in ['neg', 'negative']:
                    expanded_vars_info.append({'orig_idx': i, 'type': 'neg', 'pos_idx': None, 'neg_idx': var_idx})
                    c_expanded.append(-orig_c)
                    var_idx += 1
                else:
                    expanded_vars_info.append({'orig_idx': i, 'type': 'nonneg', 'pos_idx': var_idx, 'neg_idx': None})
                    c_expanded.append(orig_c)
                    var_idx += 1
                    
            n_structural = var_idx
            
            # 额外处理变量的上下界
            bound_constraints = []
            for i, var in enumerate(variables):
                # 预处理可能会加入 'bounds' (形式为 [low, high])
                # 或者旧版的 'upper_bound' 和 'lower_bound'
                bounds_array = var.get('bounds')
                if bounds_array is not None:
                    lb, ub = bounds_array[0], bounds_array[1]
                else:
                    ub = var.get('upper_bound')
                    lb = var.get('lower_bound')
                
                if ub is not None and ub < float('inf'):
                    coeffs = np.zeros(len(variables))
                    coeffs[i] = 1.0
                    bound_constraints.append({'type': '<=', 'coeffs': coeffs.tolist(), 'rhs': ub})
                
                # 如果非负变量，默认 lb 是 0不需额外加约束
                # 但如果是 free/neg 变过来的且有明确 lb，或 lb > 0，则加约束
                if lb is not None and lb > -float('inf'):
                    vtype = var.get('type', 'nonneg').lower()
                    if lb > 0 or vtype in ['free', 'unrestricted', 'unbounded', 'neg', 'negative']:
                        coeffs = np.zeros(len(variables))
                        coeffs[i] = 1.0
                        bound_constraints.append({'type': '>=', 'coeffs': coeffs.tolist(), 'rhs': lb})
            
            # 合并用户约束与边界约束
            all_constraints = list(constraints) + bound_constraints
            
            m = len(all_constraints)
            
            data = []
            row_idx = []
            col_idx = []
            b_vec = np.zeros(m)
            n_slack_surplus = 0
            
            # 第2步：搭建稀疏结构的 A 与 右端项 b
            for i, c_dict in enumerate(all_constraints):
                b_vec[i] = c_dict['rhs']
                orig_coeffs = c_dict['coeffs']
                
                # 填入结构变量系数
                for j, var_info in enumerate(expanded_vars_info):
                    orig_c_val = orig_coeffs[j]
                    if abs(orig_c_val) > 1e-12:
                        if var_info['type'] == 'free':
                            row_idx.extend([i, i])
                            col_idx.extend([var_info['pos_idx'], var_info['neg_idx']])
                            data.extend([orig_c_val, -orig_c_val])
                        elif var_info['type'] == 'neg':
                            row_idx.append(i)
                            col_idx.append(var_info['neg_idx'])
                            data.append(-orig_c_val)
                        else:
                            row_idx.append(i)
                            col_idx.append(var_info['pos_idx'])
                            data.append(orig_c_val)
                
                # 填入松弛与剩余变量
                ctype = c_dict['type']
                if ctype == '<=':
                    row_idx.append(i)
                    col_idx.append(n_structural + n_slack_surplus)
                    data.append(1.0)
                    n_slack_surplus += 1
                elif ctype == '>=':
                    row_idx.append(i)
                    col_idx.append(n_structural + n_slack_surplus)
                    data.append(-1.0)
                    n_slack_surplus += 1
                    
            c_full = np.zeros(n_structural + n_slack_surplus)
            c_full[:n_structural] = c_expanded
            
            A_sparse = sp.csc_matrix((data, (row_idx, col_idx)), shape=(m, n_structural + n_slack_surplus))
            
            # 第3步：进入内部稀疏引擎求解
            engine = SparseRevisedSimplex(tolerance=self.tolerance)
            is_max = objective['type'].lower() in ['max', 'maximize']
            res = engine.solve(c_full, A_sparse, b_vec, is_maximize=is_max)
            
            # 解析并提取至原变量格式
            if res['status'] == 'optimal':
                full_sol = res['solution']
                orig_sol = np.zeros(len(variables))
                
                for var_info in expanded_vars_info:
                    idx = var_info['orig_idx']
                    if var_info['type'] == 'free':
                        orig_sol[idx] = full_sol[var_info['pos_idx']] - full_sol[var_info['neg_idx']]
                    elif var_info['type'] == 'neg':
                        orig_sol[idx] = -full_sol[var_info['neg_idx']]
                    else:
                        orig_sol[idx] = full_sol[var_info['pos_idx']]
                        
                res['solution'] = orig_sol
                res['is_feasible'] = True
                res['message'] = "Sparse Revised Simplex 求解成功"
            else:
                res['is_feasible'] = False
                res['solution'] = None
                res['objective_value'] = None
                
            return res
        except Exception as e:
            print(f"❌ Revised Simplex 求解失败: {str(e)}")
            raise e
            if objective['type'].lower() in ['max', 'maximize']:
                prob = LpProblem(prob_name, LpMaximize)
            else:
                prob = LpProblem(prob_name, LpMinimize)
            
            # 创建变量
            pulp_vars = []
            for i, var in enumerate(variables):
                var_name = var.get('name', f'x{i+1}')
                var_type = var.get('type', 'nonneg').lower()
                
                # 根据变量类型设置PuLP变量类型
                if var_type == 'free':
                    # 自由变量，没有上下界
                    pulp_var = LpVariable(var_name)
                elif var_type == 'nonneg':
                    # 非负变量
                    pulp_var = LpVariable(var_name, lowBound=0)
                elif var_type == 'pos':
                    # 正变量
                    pulp_var = LpVariable(var_name, lowBound=1e-6)
                elif var_type == 'neg':
                    # 负变量
                    pulp_var = LpVariable(var_name, upBound=0)
                elif var_type == 'binary':
                    # 二元变量
                    pulp_var = LpVariable(var_name, cat='Binary')
                elif var_type == 'integer':
                    # 整数变量
                    pulp_var = LpVariable(var_name, lowBound=0, cat='Integer')
                else:
                    # 默认非负变量
                    pulp_var = LpVariable(var_name, lowBound=0)
                
                pulp_vars.append(pulp_var)
            
            # 添加目标函数
            obj_expr = sum(coeff * var for coeff, var in zip(objective['coeffs'], pulp_vars))
            prob += obj_expr
            
            # 添加约束条件
            for i, constraint in enumerate(constraints):
                ctype = constraint['type']
                coeffs = constraint['coeffs']
                rhs = constraint['rhs']
                
                # 创建约束表达式
                lhs_expr = sum(coeff * var for coeff, var in zip(coeffs, pulp_vars))
                
                # 根据约束类型添加到问题中
                if ctype == '<=':
                    prob += (lhs_expr <= rhs, f'约束{i+1}')
                elif ctype == '>=':
                    prob += (lhs_expr >= rhs, f'约束{i+1}')
                else:  # '='
                    prob += (lhs_expr == rhs, f'约束{i+1}')
            
            # 求解问题
            print("🔧 正在调用PuLP求解器...")
            prob.solve()
            
            # 解析结果
            status = prob.status
            
            if status == LpStatusOptimal:
                # 获取解
                solution = [var.varValue for var in pulp_vars]
                objective_value = prob.objective.value()
                
                print(f"✅ PuLP求解完成！")
                print(f"   状态: 最优解")
                print(f"   目标值: {objective_value:.6f}")
                
                return {
                    'status': 'optimal',
                    'solution': np.array(solution),
                    'objective_value': objective_value,
                    'message': '找到最优解'
                }
            elif status == LpStatusInfeasible:
                print(f"❌ 问题无解")
                return {
                    'status': 'infeasible',
                    'message': '问题无可行解'
                }
            elif status == LpStatusUnbounded:
                print(f"⚠️ 问题无界")
                return {
                    'status': 'unbounded',
                    'message': '问题无界'
                }
            else:
                print(f"❌ 求解失败: 状态码 {status}")
                return {
                    'status': 'error',
                    'message': f'求解失败，状态码: {status}'
                }
                
        except Exception as e:
            print(f"❌ PuLP求解过程中出错: {str(e)}")
            return {
                'status': 'error',
                'message': f'PuLP求解错误: {str(e)}'
            }
            
    def _validate_solution(self, result, objective, constraints, variables):
        """验证解"""
        if result['solution'] is None:
            return
        
        print(f"\n🔍 详细验证:")
        solution = result['solution']
        
        # 验证约束
        all_satisfied = True
        for i, constraint in enumerate(constraints):
            lhs = sum(solution[j] * constraint['coeffs'][j] for j in range(len(solution)))
            rhs = constraint['rhs']
            ctype = constraint['type']
            
            if ctype == '<=':
                satisfied = lhs <= rhs + 1e-6
                symbol = '≤'
            elif ctype == '>=':
                satisfied = lhs >= rhs - 1e-6
                symbol = '≥'
            else:  # '='
                satisfied = abs(lhs - rhs) <= 1e-6
                symbol = '='
            
            status = '✅' if satisfied else '❌'
            print(f"   {status} 约束{i+1}: {lhs:.6f} {symbol} {rhs:.6f}")
            
            if not satisfied:
                all_satisfied = False
        
        # 验证变量类型
        print(f"\n   变量验证:")
        for i, var in enumerate(variables):
            var_type = var.get('type', 'nonneg').lower()
            value = solution[i]
            
            if var_type in ['free', 'unrestricted']:
                status = '✅'  # 自由变量无限制
                print(f"   ✅ {var['name']} = {value:.6f} (自由变量)")
            elif var_type in ['nonneg', 'positive', 'pos']:
                satisfied = value >= -1e-6
                status = '✅' if satisfied else '❌'
                print(f"   {status} {var['name']} = {value:.6f} (非负)")
            elif var_type in ['neg', 'negative']:
                satisfied = value <= 1e-6
                status = '✅' if satisfied else '❌'
                print(f"   {status} {var['name']} = {value:.6f} (非正)")
            elif var_type == 'binary':
                # 检查是否接近0或1
                is_zero = abs(value) <= 1e-3
                is_one = abs(value - 1) <= 1e-3
                satisfied = is_zero or is_one
                status = '✅' if satisfied else '⚠️'
                actual_value = 0.0 if is_zero else 1.0 if is_one else value
                print(f"   {status} {var['name']} = {value:.6f} → {actual_value:.0f} (二元变量)")
            elif var_type == 'integer':
                # 检查是否接近整数
                is_integer = abs(value - round(value)) <= 1e-3
                satisfied = True  # 整数变量在LP松弛中可能不是整数，这里只做提示
                status = '✅' if is_integer else '⚠️'
                actual_value = round(value) if is_integer else value
                print(f"   {status} {var['name']} = {value:.6f} → {actual_value:.0f} (整数变量)")
        
        print(f"\n   📊 解的可行性: {'✅ 可行' if all_satisfied else '❌ 不可行'}")


# 便捷函数
def solve_complex_lp(objective, constraints, variables):
    """求解复杂线性规划问题"""
    solver = UltimateLPSolver()
    return solver.solve(objective, constraints, variables)
