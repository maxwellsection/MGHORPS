import re
import numpy as np
from typing import Dict, List, Any

class LingoCompiler:
    """
    仿 LINGO 操作逻辑的代数建模预编译器。
    将人类可读的代数文本（如 MAX = 20*x1 + 30*x2;）编译为 standard_form 字典。
    """
    def __init__(self):
        self.variables = []
        self.var_name_to_idx = {}
        self.objective = {'type': 'max', 'coeffs': []}
        self.constraints = []
        
    def _get_or_create_var(self, name: str) -> int:
        if name not in self.var_name_to_idx:
            idx = len(self.variables)
            self.var_name_to_idx[name] = idx
            self.variables.append({'name': name, 'type': 'nonneg'})  # Default to nonneg
        return self.var_name_to_idx[name]

    def _parse_expression(self, exprStr: str) -> Dict[str, float]:
        """
        解析单边的线性表达式，如 "- 3*X1 + 2 * X2 - X3 + 5"
        返回 变量名->系数 的映射，以及常数项 '_constant_'
        """
        exprStr = exprStr.replace(' ', '')
        if not exprStr:
            return {'_constant_': 0.0}
            
        # 处理开头的可选正负号
        if exprStr[0] not in ('+', '-'):
            exprStr = '+' + exprStr
            
        tokens = re.findall(r'([+-])([^+-]+)', exprStr)
        
        parsed = {'_constant_': 0.0}
        
        for sign, term in tokens:
            coef_multiplier = 1.0 if sign == '+' else -1.0
            
            # 例如 "20*X1", "X1", "5.5"
            term_split = term.split('*')
            if len(term_split) == 2:
                # coef * var
                try:
                    c_val = float(term_split[0])
                    v_name = term_split[1]
                except ValueError: # var * coef
                    v_name = term_split[0]
                    c_val = float(term_split[1])
                parsed[v_name] = parsed.get(v_name, 0.0) + coef_multiplier * c_val
                self._get_or_create_var(v_name)
            elif len(term_split) == 1:
                # 可能是存数字，或者纯变量
                val = term_split[0]
                try:
                    c_val = float(val)
                    parsed['_constant_'] += coef_multiplier * c_val
                except ValueError:
                    # 它是纯变量，系数为 1
                    parsed[val] = parsed.get(val, 0.0) + coef_multiplier * 1.0
                    self._get_or_create_var(val)
                    
        return parsed

    def compile(self, text: str) -> Dict[str, Any]:
        """
        将整段文本编译为终极线性规划器可识别的标准字典。
        语法约束：
        语句以分号 ; 结束。
        第一句通常是 MAX = ...; 或 MIN = ...;
        包含可选的约束关系 <=, >=, =
        特定变量标识： @FREE(x); @BND(10, x, 50);
        """
        self.variables = []
        self.var_name_to_idx = {}
        self.objective = {'type': 'max', 'coeffs': []}
        self.constraints = []
        
        # 移除注释 (支持 ! ... ;)
        text = re.sub(r'!.*?;', ';', text, flags=re.DOTALL)
        
        # 按分号分割为语句
        statements = [stmt.strip() for stmt in text.split(';') if stmt.strip()]
        
        for stmt in statements:
            # 解析特征标识符号
            if stmt.upper().startswith('@FREE'):
                v_match = re.search(r'@FREE\s*\((.*?)\)', stmt, re.IGNORECASE)
                if v_match:
                    v_name = v_match.group(1).strip()
                    self._get_or_create_var(v_name)
                    self.variables[self.var_name_to_idx[v_name]]['type'] = 'free'
                continue
            elif stmt.upper().startswith('@BND'):
                v_match = re.search(r'@BND\s*\((.*?),(.*?),(.*?)\)', stmt, re.IGNORECASE)
                if v_match:
                    lb = float(v_match.group(1).strip())
                    v_name = v_match.group(2).strip()
                    ub = float(v_match.group(3).strip())
                    self._get_or_create_var(v_name)
                    self.variables[self.var_name_to_idx[v_name]]['bounds'] = [lb, ub]
                continue
            
            # 解析 目标函数 MAX / MIN
            if stmt.upper().startswith('MAX') or stmt.upper().startswith('MIN'):
                if '=' in stmt:
                    obj_type, expr = stmt.split('=', 1)
                    obj_type = obj_type.strip().lower() # 'max' or 'min'
                    if 'max' in obj_type:
                        self.objective['type'] = 'max'
                    else:
                        self.objective['type'] = 'min'
                    
                    parsed_expr = self._parse_expression(expr)
                    # 我们暂时将目标函数存储在字典内，等所有变量扫描完毕再转数组
                    self.objective['expr_dict'] = parsed_expr
                continue
            
            # 解析SUBJECT TO标识符 (仅作语法跳过)
            if stmt.upper() == 'SUBJECT TO' or stmt.upper() == 'ST':
                continue
                
            # 解析普通线性约束
            match = re.search(r'(.*?)(<=|>=|=|<|>)(.*)', stmt)
            if match:
                lhs_str = match.group(1).strip()
                operator = match.group(2).strip()
                rhs_str = match.group(3).strip()
                
                # 简单处理 < 或者 > 转换为 <=, >=
                if operator == '<': operator = '<='
                if operator == '>': operator = '>='
                
                lhs_parsed = self._parse_expression(lhs_str)
                rhs_parsed = self._parse_expression(rhs_str)
                
                # 统一移项到左边: lhs - rhs <= 0 
                # (lhs_coef - rhs_coef) * X <= rhs_const - lhs_const
                merged = {}
                for k, v in lhs_parsed.items():
                    merged[k] = merged.get(k, 0.0) + v
                for k, v in rhs_parsed.items():
                    merged[k] = merged.get(k, 0.0) - v
                
                rhs_val = -(merged.get('_constant_', 0.0))
                if '_constant_' in merged:
                    del merged['_constant_']
                    
                self.constraints.append({
                    'type': operator,
                    'expr_dict': merged,
                    'rhs': rhs_val,
                    'original': stmt
                })

        # 后处理：统一所有的变量系数向量
        n_vars = len(self.variables)
        
        # 组装 objective
        obj_coeffs = [0.0] * n_vars
        obj_expr = self.objective.get('expr_dict', {})
        for k, v in obj_expr.items():
            if k != '_constant_' and k in self.var_name_to_idx:
                obj_coeffs[self.var_name_to_idx[k]] = v
        self.objective['coeffs'] = obj_coeffs
        if 'expr_dict' in self.objective:
            del self.objective['expr_dict']
            
        # 组装 constraints
        final_constraints = []
        for c in self.constraints:
            c_coeffs = [0.0] * n_vars
            for k, v in c['expr_dict'].items():
                if k in self.var_name_to_idx:
                    c_coeffs[self.var_name_to_idx[k]] = v
            final_constraints.append({
                'type': c['type'],
                'coeffs': c_coeffs,
                'rhs': c['rhs']
            })
            
        return {
            'objective': self.objective,
            'constraints': final_constraints,
            'variables': self.variables
        }

if __name__ == "__main__":
    # 测试一下基础编译器
    model_text = """
    ! 这是一个标准的线性规划测试 ;
    MAX = 20 * X1 + 30 * X2;
    SUBJECT TO;
    X1 + X2 <= 50;
    3 * X1 + 2 * X2 <= 100;
    @FREE(X1);
    @BND(10, X2, 30);
    """
    
    compiler = LingoCompiler()
    res = compiler.compile(model_text)
    print("变量表:", res['variables'])
    print("目标函数:", res['objective'])
    print("约束个数:", len(res['constraints']))
    for c in res['constraints']:
        print(f" {c['coeffs']} {c['type']} {c['rhs']}")
