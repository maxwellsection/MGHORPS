import numpy as np
from typing import Dict, List, Any, Tuple

class AdvancedPresolver:
    """
    Advanced Presolve engine for Linear Programming.
    Simplifies the problem by removing redundant constraints, fixing variables,
    and substituting single-variable equalities.
    """
    
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        
        # State for postsolve recovery
        self.fixed_vars: Dict[str, float] = {}
        self.substituted_vars: Dict[str, Dict[str, Any]] = {}
        self.original_problem: Dict[str, Any] = None
        self.var_idx_map: Dict[str, int] = {}
        
    def presolve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for presolving.
        """
        print("\n🔍 启动高级预处理引擎 (Advanced Presolve Engine) ...")
        self.original_problem = problem
        
        # We work with a copy that we will mutate
        p = {}
        for k, v in problem.items():
            if k == 'objective':
                p[k] = {'type': problem['objective']['type'], 'coeffs': problem['objective']['coeffs'].copy()}
            elif k == 'constraints':
                p[k] = [c.copy() for c in problem['constraints']]
            elif k == 'variables':
                p[k] = [v.copy() for v in problem['variables']]
            else:
                try:
                    p[k] = v.copy() if hasattr(v, 'copy') else v
                except Exception:
                    p[k] = v
        
        # Create mapping var_name -> idx
        self.var_idx_map = {v['name']: i for i, v in enumerate(p['variables'])}
        
        n_orig_vars = len(p['variables'])
        n_orig_cons = len(p['constraints'])
        
        iteration = 0
        changed = True
        
        while changed and iteration < 5: # Max 5 passes
            changed = False
            iteration += 1
            
            # 1. Update Bounds & Fix Variables
            changed |= self._fix_variables(p)
            
            # 2. Singleton Rows Elimination
            changed |= self._eliminate_singleton_rows(p)
            
            # 3. Empty Rows/Cols Elimination
            changed |= self._remove_empty_rows(p)
        
        # Re-pack the problem (remove fixed vars from objective & constraints)
        final_p = self._pack_problem(p)
        
        n_final_vars = len(final_p['variables'])
        n_final_cons = len(final_p['constraints'])
        
        print(f"✅ 预处理完成 (经过 {iteration} 轮迭代):")
        print(f"   - 变量: {n_orig_vars} -> {n_final_vars} (移除了 {n_orig_vars - n_final_vars} 个)")
        print(f"   - 约束: {n_orig_cons} -> {n_final_cons} (移除了 {n_orig_cons - n_final_cons} 个)")
        print(f"   - 已固定变量: {len(self.fixed_vars)} 个")
        
        return final_p
        
    def _fix_variables(self, p: Dict[str, Any]) -> bool:
        """Fix variables whose lower bound equals upper bound."""
        changed = False
        
        for v in p['variables']:
            name = v['name']
            if name in self.fixed_vars:
                continue
                
            bounds = v.get('bounds', [0.0, float('inf')])
            if abs(bounds[1] - bounds[0]) < self.tolerance:
                val = bounds[0]
                self.fixed_vars[name] = val
                changed = True
                print(f"   🔒 固定变量: {name} = {val}")
                
        return changed
        
    def _eliminate_singleton_rows(self, p: Dict[str, Any]) -> bool:
        """
        Find constraints with exactly one non-zero coefficient (singleton rows).
        a_j * x_j <= b  -->  update bound on x_j
        """
        changed = False
        constraints_to_remove = []
        
        for i, c in enumerate(p['constraints']):
            coeffs = c['coeffs']
            
            # Count non-zeros that aren't fixed
            non_zeros = []
            for j, val in enumerate(coeffs):
                var_name = p['variables'][j]['name']
                if abs(val) > self.tolerance and var_name not in self.fixed_vars:
                    non_zeros.append((j, var_name, val))
                    
            if len(non_zeros) == 1:
                col_idx, var_name, a_j = non_zeros[0]
                rhs = c['rhs']
                
                # Adjust RHS for fixed variables present in this constraint
                for j, val in enumerate(coeffs):
                    v_n = p['variables'][j]['name']
                    if j != col_idx and v_n in self.fixed_vars:
                        rhs -= val * self.fixed_vars[v_n]
                
                bnd_val = rhs / a_j
                sense = c['type']
                
                # Convert constraint to variable bound
                var_bounds = p['variables'][col_idx].get('bounds', [0.0, float('inf')])
                current_low, current_high = var_bounds[0], var_bounds[1]
                
                updated = False
                if sense == '=':
                    # x_j = bnd_val
                    p['variables'][col_idx]['bounds'] = [bnd_val, bnd_val]
                    updated = True
                elif (sense == '<=' and a_j > 0) or (sense == '>=' and a_j < 0):
                    # x_j <= bnd_val
                    if bnd_val < current_high:
                        p['variables'][col_idx]['bounds'] = [current_low, bnd_val]
                        updated = True
                elif (sense == '>=' and a_j > 0) or (sense == '<=' and a_j < 0):
                    # x_j >= bnd_val
                    if bnd_val > current_low:
                        p['variables'][col_idx]['bounds'] = [bnd_val, current_high]
                        updated = True
                        
                if updated:
                    constraints_to_remove.append(i)
                    changed = True
                    print(f"   ✂️ 将单变量约束转化为边界: {var_name} ({sense} {bnd_val:.4f})")
                    
        # Remove processed constraints
        for i in sorted(constraints_to_remove, reverse=True):
            del p['constraints'][i]
            
        return changed
        
    def _remove_empty_rows(self, p: Dict[str, Any]) -> bool:
        """Remove constraints where all active variable coefficients are zero."""
        changed = False
        constraints_to_remove = []
        
        for i, c in enumerate(p['constraints']):
            coeffs = c['coeffs']
            rhs = c['rhs']
            
            is_empty = True
            adjusted_rhs = rhs
            
            for j, val in enumerate(coeffs):
                var_name = p['variables'][j]['name']
                if abs(val) > self.tolerance:
                    if var_name in self.fixed_vars:
                        adjusted_rhs -= val * self.fixed_vars[var_name]
                    else:
                        is_empty = False
                        break
                        
            if is_empty:
                # Check feasibility
                sense = c['type']
                feasible = False
                if sense == '<=' and -self.tolerance <= adjusted_rhs:
                    feasible = True
                elif sense == '>=' and self.tolerance >= adjusted_rhs:
                    feasible = True
                elif sense == '=' and abs(adjusted_rhs) <= self.tolerance:
                    feasible = True
                    
                if not feasible:
                    print(f"   ❌ 警告: 发现不可行的绝对矛盾约束: 0 {sense} {adjusted_rhs}")
                    # Usually we'd raise an exception, but let the solver handle the infeasibility
                    # (it will naturally fail or report infeasible)
                else:
                    constraints_to_remove.append(i)
                    changed = True
                    
        for i in sorted(constraints_to_remove, reverse=True):
            del p['constraints'][i]
            
        return changed
        
    def _pack_problem(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds the final simplified problem dictionary by removing
        fixed variables from the objective and constraints.
        """
        final_vars = []
        keep_idxs = []
        
        # 1. Filter variables
        for i, v in enumerate(p['variables']):
            if v['name'] not in self.fixed_vars:
                final_vars.append(v)
                keep_idxs.append(i)
                
        if len(final_vars) == len(p['variables']):
            return p # No structural changes
            
        # 2. Rebuild Objective
        obj_val_offset = 0.0
        orig_obj_coeffs = p['objective']['coeffs']
        final_obj_coeffs = []
        
        for i, val in enumerate(orig_obj_coeffs):
            var_name = p['variables'][i]['name']
            if var_name in self.fixed_vars:
                obj_val_offset += val * self.fixed_vars[var_name]
            else:
                final_obj_coeffs.append(val)
                
        # Keep other original keys (like variable_types from standardization phase)
        final_p = {}
        for k, v in p.items():
            if k not in ['objective', 'variables', 'constraints']:
                final_p[k] = v
                
        final_p['objective'] = {
            'type': p['objective']['type'],
            'coeffs': final_obj_coeffs,
            'offset': obj_val_offset # Store offset for postsolve
        }
        final_p['variables'] = final_vars
        final_p['constraints'] = []
        
        # 3. Rebuild Constraints
        for c in p['constraints']:
            orig_coeffs = c['coeffs']
            final_coeffs = []
            rhs = c['rhs']
            
            for i, val in enumerate(orig_coeffs):
                var_name = p['variables'][i]['name']
                if var_name in self.fixed_vars:
                    rhs -= val * self.fixed_vars[var_name]
                else:
                    final_coeffs.append(val)
                    
            final_p['constraints'].append({
                'name': c.get('name', ''),
                'type': c['type'],
                'coeffs': final_coeffs,
                'rhs': rhs
            })
            
        return final_p
        
    def postsolve(self, presolved_solution: List[float], presolved_vars: List[Dict]) -> List[float]:
        """
        Reconstructs the full solution space including fixed and eliminated variables.
        """
        full_solution = [0.0] * len(self.original_problem['variables'])
        
        # 1. Map returned solution to active variables
        for i, val in enumerate(presolved_solution):
            var_name = presolved_vars[i]['name']
            orig_idx = self.var_idx_map[var_name]
            full_solution[orig_idx] = val
            
        # 2. Fill in fixed variables
        for var_name, val in self.fixed_vars.items():
            orig_idx = self.var_idx_map[var_name]
            full_solution[orig_idx] = val
            
        # (Future: Handle substituted equalities here by back-substituting)
        
        return full_solution
