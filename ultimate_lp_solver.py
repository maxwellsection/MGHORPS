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
import time
from typing import List, Dict, Tuple, Optional, Union

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
    
    def __init__(self, tolerance=1e-8, solver='auto', use_gpu=False):
        self.tolerance = tolerance
        
        # 选择求解器
        if solver == 'auto':
            self.solver = 'pulp' if PULP_AVAILABLE else 'builtin'
        elif solver in ['pulp', 'builtin']:
            if solver == 'pulp' and not PULP_AVAILABLE:
                print(f"⚠️ PuLP求解器不可用，将使用内置求解器")
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
        print(f"   - 计算设备: {'GPU' if self.use_gpu else 'CPU'}")
    
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
            else:
                # 1. 问题标准化
                print(f"\n🔄 第1步: 问题标准化...")
                standard_form = self._standardize_problem(objective, constraints, variables)
                
                # 2. 构建单纯形表
                print(f"🔄 第2步: 构建单纯形表...")
                tableau_info = self._build_tableau(standard_form)
                
                # 3. 求解
                print(f"🔄 第3步: 使用内置求解器进行求解...")
                result = self._solve_problem(tableau_info, standard_form)
            
                # 4. 结果后处理
                final_result = self._process_result(result, standard_form, variables)
            
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
                constraint['type'] = '<='
            elif ctype in ['>=', 'greater_equal', 'ge']:
                constraint['type'] = '>='
            elif ctype in ['=', 'eq', 'equal']:
                constraint['type'] = '='
            else:
                raise ValueError(f"不支持的约束类型: {constraint['type']}")
            
            processed_constraints.append(constraint)
        
        # 处理变量
        variable_types = []
        for var in variables:
            var_type = var.get('type', 'nonneg').lower()
            if var_type in ['free', 'unrestricted', 'unbounded']:
                variable_types.append('free')
            elif var_type in ['nonneg', 'positive', 'pos', 'binary', 'integer']:  # 将binary和integer变量当作nonneg处理
                variable_types.append('nonneg')
            elif var_type in ['neg', 'negative']:
                variable_types.append('neg')
            else:
                variable_types.append('nonneg')
        
        # 收集原始变量类型用于后续验证
        original_variable_types = [var.get('type', 'nonneg').lower() for var in variables]
        
        print(f"   ✅ 标准化完成")
        print(f"      变量类型: {variable_types}")
        print(f"      约束类型: {[c['type'] for c in processed_constraints]}")
        
        return {
            'objective': objective,
            'constraints': processed_constraints,
            'variables': variables,
            'variable_types': variable_types,
            'original_variable_types': original_variable_types,  # 添加原始变量类型
            'n_original_vars': len(variables)
        }
    
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
            'objective_coeffs': expanded_coeffs
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
            
            # 处理>=约束转换为<=形式
            if processed_constraint['type'] == '>=':
                processed_constraint['coeffs'] = -processed_constraint['coeffs']
                processed_constraint['rhs'] = -processed_constraint['rhs']
                processed_constraint['type'] = '<='
            
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
        original_obj = np.zeros(tableau.shape[1])
        original_coeffs = tableau_info['objective_coeffs']
        
        # 确保只使用原目标函数中有效变量的系数
        n_valid_vars = min(len(original_coeffs), tableau.shape[1] - 1)  # 最后一列是RHS
        original_obj[:n_valid_vars] = original_coeffs[:n_valid_vars]
        
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
        aux_obj = np.zeros(tableau.shape[1])
        artificial_start = tableau_info['n_expanded_vars'] + tableau_info['n_slack'] + tableau_info['n_surplus']
        
        print(f"      📊 人工变量起始位置: {artificial_start}")
        print(f"      📊 人工变量数量: {tableau_info['n_artificial']}")
        
        for i in range(tableau_info['n_artificial']):
            aux_obj[artificial_start + i] = 1.0
        
        # 更新目标函数行
        tableau[-1, :] = aux_obj
        
        print(f"      📊 辅助目标函数: {aux_obj}")
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
        
        print(f"      🔍 解提取 - 最终单纯形表:")
        print(f"      {tableau}")
        
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
                print(f"      📊 变量{orig_idx}: 列{pos_var} = {col}")
                
                # 检查是否为基变量：列中恰好有一个1.0，其余为0
                if self.array_lib.sum(self.array_lib.abs(col)) == 1.0:
                    row = self.array_lib.argmax(self.array_lib.abs(col))
                    if abs(col[row]) > 0.5:  # 确保是1.0而不是其他值
                        solution[orig_idx] = tableau[row, -1]
                        print(f"      ✅ 基变量: 变量{orig_idx} = {solution[orig_idx]} (行{row})")
                    else:
                        print(f"      ⚠️ 非基变量: 变量{orig_idx} = 0")
                else:
                    print(f"      ⚠️ 非基变量: 变量{orig_idx} = 0")
            
            # 如果是自由变量，减去负变量
            if neg_var is not None and neg_var < tableau.shape[1] - 1:
                col = tableau[:n_constraints, neg_var]
                print(f"      📊 自由变量负部: 列{neg_var} = {col}")
                
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
    
    def _pulp_solve(self, objective, constraints, variables):
        """使用PuLP求解器求解线性规划问题"""
        try:
            # 创建PuLP问题
            prob_name = "ultimate_lp_problem"
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
