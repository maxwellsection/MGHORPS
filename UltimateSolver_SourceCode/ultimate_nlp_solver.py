import numpy as np
import warnings
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

class UltimateNLPSolver:
    """
    非线性规划求解器 (包装 scipy.optimize.minimize)
    处理由 lingo_compiler.py 生成的非线性表达式字符串
    """
    def __init__(self, verbose_options=None):
        self.verbose_options = verbose_options or {}

    def solve(self, objective, constraints, variables):
        n_vars = len(variables)
        
        # 构建变量边界
        lower_bounds = []
        upper_bounds = []
        for var in variables:
            v_type = var.get('type', 'nonneg')
            bnds = var.get('bounds', None)
            
            lb, ub = 0.0, np.inf
            if v_type == 'free':
                lb = -np.inf
            elif v_type == 'neg':
                lb, ub = -np.inf, 0.0
                
            if bnds:
                if bnds[0] is not None: lb = bnds[0]
                if bnds[1] is not None: ub = bnds[1]
                
            lower_bounds.append(lb)
            upper_bounds.append(ub)
            
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # 构建目标函数
        obj_eval_str = objective.get('nlp_expr', '')
        if not obj_eval_str:
            # Fallback to linear
            coeffs = np.array(objective.get('coeffs', [0]*n_vars))
            def obj_func(x):
                return np.dot(coeffs, x)
        else:
            # Ensure safe eval environment
            safe_dict = {'np': np, 'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos}
            def obj_func(x):
                try:
                    # x 必须作为局部变量传入使得 eval('x[0] + exp(x[1])') 能够计算
                    return eval(obj_eval_str, safe_dict, {'x': x})
                except Exception as e:
                    return 1e9 # Penalty on error
                    
        # 处理最大化
        sign = -1.0 if objective.get('type', 'min') in ['max', 'maximize'] else 1.0
        def final_obj(x):
            return sign * obj_func(x)
            
        # 构建约束
        scipy_constraints = []
        for i, c in enumerate(constraints):
            ctype = c.get('type', '<=')
            rhs = float(c.get('rhs', 0.0))
            
            if c.get('is_nonlinear', False):
                eval_str = c['nlp_expr']
                safe_dict = {'np': np, 'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos}
                
                # 闭包捕获 eval_str
                def make_nl_cons(expr_str):
                    return lambda x: eval(expr_str, safe_dict, {'x': x})
                    
                con_func = make_nl_cons(eval_str)
                
                if ctype == '<=':
                    scipy_constraints.append(NonlinearConstraint(con_func, -np.inf, rhs))
                elif ctype == '>=':
                    scipy_constraints.append(NonlinearConstraint(con_func, rhs, np.inf))
                else: # ==
                    scipy_constraints.append(NonlinearConstraint(con_func, rhs, rhs))
            else:
                # 线性约束
                coeffs = c.get('coeffs', [0.0]*n_vars)
                if ctype == '<=':
                    scipy_constraints.append(LinearConstraint(coeffs, -np.inf, rhs))
                elif ctype == '>=':
                    scipy_constraints.append(LinearConstraint(coeffs, rhs, np.inf))
                else:
                    scipy_constraints.append(LinearConstraint(coeffs, rhs, rhs))
                    
        # 初始解规划
        x0 = np.zeros(n_vars)
        for i in range(n_vars):
            if lower_bounds[i] > 0 and lower_bounds[i] != np.inf:
                x0[i] = lower_bounds[i]
            elif upper_bounds[i] < 0 and upper_bounds[i] != -np.inf:
                x0[i] = upper_bounds[i]
                
        # 求解
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(final_obj, x0, method='SLSQP', bounds=bounds, constraints=scipy_constraints)
            
        final_result = {
            'status': 'optimal' if res.success else 'infeasible',
            'message': res.message,
            'solution': res.x,
            'objective_value': sign * res.fun,
            'is_feasible': res.success,
            'solve_time': 0.0,
            'solver': 'scipy_nlp'
        }
        return final_result
