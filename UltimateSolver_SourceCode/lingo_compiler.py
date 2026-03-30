import re
import numpy as np
from typing import Dict, List, Any, Tuple

class LingoCompiler:
    """
    仿 LINGO 操作逻辑的代数建模预编译器。
    支持 SETS, DATA, @FOR, @SUM, @MIN, @MAX, @EXP, @LOG, 及基础代数表达式。
    将人类可读的代数文本编译为标准字典供终极求解器使用。
    """
    def __init__(self):
        self.variables = []
        self.var_name_to_idx = {}
        self.objective = {'type': 'max', 'coeffs': []}
        self.constraints = []
        self.sets = {}
        self.data = {}
        self.verbose_options = {}

    def _get_or_create_var(self, name: str) -> int:
        if name not in self.var_name_to_idx:
            idx = len(self.variables)
            self.var_name_to_idx[name] = idx
            self.variables.append({'name': name, 'type': 'nonneg'})
        return self.var_name_to_idx[name]

    def _parse_sets(self, text: str):
        """解析 SETS: ... ENDSETS 块"""
        pattern = re.compile(r'SETS:(.*?)ENDSETS', re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match: return text
        
        sets_body = match.group(1)
        # 例如 CITIES /1..5/: DEMAND, SUPPLY;
        stmt_pattern = re.compile(r'(\w+)\s*(?:/\s*(.*?)\s*/)?\s*(?::\s*(.*?))?;')
        for m in stmt_pattern.finditer(sets_body):
            set_name = m.group(1).strip()
            elements_str = m.group(2)
            attrs_str = m.group(3)
            
            elements = []
            if elements_str:
                if '..' in elements_str:
                    start, end = elements_str.split('..')
                    try:
                        elements = [str(i) for i in range(int(start), int(end)+1)]
                    except:
                        pass # Handle non-integer ranges if needed
                else:
                    elements = [e.strip() for e in elements_str.replace(',', ' ').split() if e.strip()]
            
            attributes = []
            if attrs_str:
                attributes = [a.strip() for a in attrs_str.split(',') if a.strip()]
                
            self.sets[set_name] = {'elements': elements, 'attributes': attributes}
            
        return pattern.sub('', text)

    def _parse_data(self, text: str):
        """解析 DATA: ... ENDDATA 块"""
        pattern = re.compile(r'DATA:(.*?)ENDDATA', re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match: return text
        
        data_body = match.group(1)
        # 每行一个 DEMAND = 10 20 30;
        stmt_pattern = re.compile(r'(\w+)\s*=\s*(.*?);', re.DOTALL)
        for m in stmt_pattern.finditer(data_body):
            attr_name = m.group(1).strip()
            values_str = m.group(2)
            values = [float(v.strip()) for v in values_str.replace(',', ' ').split() if v.strip()]
            self.data[attr_name] = values
            
        return pattern.sub('', text)

    def _expand_expression(self, exprStr: str, loop_vars: dict) -> str:
        """展开含有循环变量的表达式，如 X(i) 替换为 X_1"""
        # 首先替换数据常量
        for attr, values in self.data.items():
            # 找到类似于 DEMAND(i) 的模式
            # 为了简化，我们假设集合的索引被替换成具体的数字
            pass
        
        # 替换循环变量, 将 X(i) 中的 i 替换为它的值
        res = exprStr
        for var, val in loop_vars.items():
            # 使用正则替换完整的词
            # \b var \b 未必可行，因为 i 可能在括号内
            # 我们查找形如 (i) 或者 [i] 并在展开变量名时下划线连接，如 X_1
            # 这里做个简单的正则，把 (i) 换成 _val
            pattern = re.compile(r'\(\s*' + re.escape(var) + r'\s*\)')
            res = pattern.sub(f'_{val}', res)
            
        # 然后把数据项直接替换成数值
        for attr, values in self.data.items():
            # 如果展开后是 DEMAND_1，我们需要找到它的索引
            # 假设 elements 是 1..N，那么 DEMAND_1 对应 values[0]
            # 为了灵活性，我们在这里寻找 attributes
            for set_name, set_info in self.sets.items():
                if attr in set_info['attributes']:
                    for idx, elem in enumerate(set_info['elements']):
                        old_str = f"{attr}_{elem}"
                        if old_str in res and idx < len(values):
                            res = res.replace(old_str, str(values[idx]))
        return res

    def _extract_blocks(self, text: str, keyword: str):
        """通用的大括号/小括号平衡提取器。提取 keyword(...) 形式。"""
        results = []
        idx = 0
        upper_text = text.upper()
        kw = keyword.upper() + "("
        while True:
            start_idx = upper_text.find(kw, idx)
            if start_idx == -1:
                # 尝试 keyword (...)
                kw_space = keyword.upper() + " ("
                start_idx_2 = upper_text.find(kw_space, idx)
                if start_idx_2 == -1:
                    break
                else:
                    start_idx = start_idx_2
                    paren_start = start_idx + len(keyword) + 1
            else:
                paren_start = start_idx + len(keyword)
                
            # 从 paren_start 开始找匹配的右括号
            open_count = 1
            curr_idx = paren_start + 1
            while curr_idx < len(text) and open_count > 0:
                if text[curr_idx] == '(':
                    open_count += 1
                elif text[curr_idx] == ')':
                    open_count -= 1
                curr_idx += 1
                
            if open_count == 0:
                # 提取到了内容
                outer_str = text[start_idx:curr_idx]
            if open_count == 0:
                # 提取到了内容
                outer_str = text[start_idx:curr_idx]
                inner_str = text[paren_start+1:curr_idx-1]
                results.append((start_idx, curr_idx, outer_str, inner_str))
                idx = curr_idx
            else:
                break
        return results

    def _eval_condition(self, cond_exprStr: str, loop_vars: dict) -> bool:
        """评估 LINGO 的逻辑条件 (例如 iterator #GT# 3)"""
        expr = self._expand_expression(cond_exprStr, loop_vars)
        for var, val in loop_vars.items():
            pattern = re.compile(r'\b' + re.escape(var) + r'\b')
            expr = pattern.sub(str(val), expr)
            
        expr = expr.upper()
        expr = expr.replace('#EQ#', '==').replace('#NE#', '!=')
        expr = expr.replace('#GT#', '>').replace('#GE#', '>=')
        expr = expr.replace('#LT#', '<').replace('#LE#', '<=')
        expr = expr.replace('#AND#', ' and ').replace('#OR#', ' or ').replace('#NOT#', ' not ')
        try:
            return bool(eval(expr))
        except:
            return True

    def _resolve_aggregators(self, text: str) -> str:
        """处理 @SUM, @MAX, @MIN 宏展开，支持条件过滤"""
        for kw in ["@SUM", "@MAX", "@MIN"]:
            while True:
                blocks = self._extract_blocks(text, kw)
                if not blocks: break
                
                start, end, outer, inner = blocks[0]
                parts = inner.split(':', 1)
                if len(parts) != 2:
                    text = text[:start] + " " + text[end:]
                    continue
                    
                decl_full = parts[0].strip()
                inner_expr = parts[1].strip()
                
                cond_expr = None
                if '|' in decl_full:
                    decl_parts = decl_full.split('|', 1)
                    iterator_decl = decl_parts[0].strip()
                    cond_expr = decl_parts[1].strip()
                else:
                    iterator_decl = decl_full
                
                m = re.match(r'(\w+)\s*\(\s*(\w+)\s*\)', iterator_decl)
                if not m:
                    text = text[:start] + " " + text[end:]
                    continue
                    
                set_name = m.group(1)
                iterator = m.group(2)
                
                if set_name not in self.sets:
                    raise ValueError(f"Unknown set in {kw}: {set_name}")
                    
                elements = self.sets[set_name]['elements']
                expanded_terms = []
                
                for elem in elements:
                    if cond_expr and not self._eval_condition(cond_expr, {iterator: elem}):
                        continue
                    term = self._expand_expression(inner_expr, {iterator: elem})
                    expanded_terms.append(term)
                    
                if kw == "@SUM":
                    replacement = " + ".join(expanded_terms) if expanded_terms else " 0 "
                elif kw == "@MAX":
                    try:
                        # 尝试预先求值常数聚合
                        replacement = str(max([float(eval(t)) for t in expanded_terms])) if expanded_terms else "-1e9"
                    except:
                        replacement = "max(" + ",".join(expanded_terms) + ")" if expanded_terms else "0"
                elif kw == "@MIN":
                    try:
                        replacement = str(min([float(eval(t)) for t in expanded_terms])) if expanded_terms else "1e9"
                    except:
                        replacement = "min(" + ",".join(expanded_terms) + ")" if expanded_terms else "0"
                    
                text = text[:start] + " " + replacement + " " + text[end:]
                
        return text

    def _is_nlp(self, text: str) -> bool:
        kwds = ['@EXP', '@LOG', '@SIN', '@COS', '@TAN', '^', '*']
        up_text = text.upper()
        # '*' is ambiguous (can be linear 2*x) but if it's var*var it's NLP.
        for kw in kwds[:5]:
            if kw in up_text: return True
        # Check for var*var or var^2, simplifying: we just check for @ keywords for now
        return False

    def _build_nlp_expr(self, expr_str: str) -> str:
        res = expr_str
        # Replace mathematical functions
        res = re.sub(r'@EXP\s*\(', 'np.exp(', res, flags=re.IGNORECASE)
        res = re.sub(r'@LOG\s*\(', 'np.log(', res, flags=re.IGNORECASE)
        res = re.sub(r'@SIN\s*\(', 'np.sin(', res, flags=re.IGNORECASE)
        res = re.sub(r'@COS\s*\(', 'np.cos(', res, flags=re.IGNORECASE)
        res = re.sub(r'@TAN\s*\(', 'np.tan(', res, flags=re.IGNORECASE)
        
        # We need to map variables to x[idx]. 
        # But indices are defined globally.
        # We will do variable replacement in a final pass after all variables are known!
        # So here we just return the math-replaced string.
        return res

    def _parse_expression(self, exprStr: str) -> Dict[str, float]:
        """解析单边线性表达式，同之前，带有简单的常量合并和变量提取"""
        exprStr = exprStr.replace(' ', '')
        if not exprStr:
            return {'_constant_': 0.0}
            
        if exprStr[0] not in ('+', '-'):
            exprStr = '+' + exprStr
            
        tokens = re.findall(r'([+-])([^+-]+)', exprStr)
        parsed = {'_constant_': 0.0}
        
        for sign, term in tokens:
            coef_multiplier = 1.0 if sign == '+' else -1.0
            
            # 支持数学函数 @EXP, @LOG, @SIN 等（非线性）
            # 注意目前的后端只能解线性规划。暂将其解析为特殊标记。
            # 这里简单支持线性的合并
            
            term_split = term.split('*')
            if len(term_split) == 2:
                try:
                    c_val = float(term_split[0])
                    v_name = term_split[1]
                except ValueError:
                    v_name = term_split[0]
                    c_val = float(term_split[1])
                parsed[v_name] = parsed.get(v_name, 0.0) + coef_multiplier * c_val
                self._get_or_create_var(v_name)
            elif len(term_split) == 1:
                val = term_split[0]
                try:
                    c_val = float(val)
                    parsed['_constant_'] += coef_multiplier * c_val
                except ValueError:
                    parsed[val] = parsed.get(val, 0.0) + coef_multiplier * 1.0
                    self._get_or_create_var(val)
                    
        return parsed

    def _process_statement(self, stmt: str):
        """处理单条语句"""
        stmt = stmt.strip()
        if not stmt: return
        
        # 处理变量修饰
        if stmt.upper().startswith('@FREE'):
            v_match = re.search(r'@FREE\s*\((.*?)\)', stmt, re.IGNORECASE)
            if v_match:
                v_name = v_match.group(1).strip()
                self._get_or_create_var(v_name)
                self.variables[self.var_name_to_idx[v_name]]['type'] = 'free'
            return
            
        elif stmt.upper().startswith('@BND'):
            v_match = re.search(r'@BND\s*\((.*?),(.*?),(.*?)\)', stmt, re.IGNORECASE)
            if v_match:
                lb = float(v_match.group(1).strip())
                v_name = v_match.group(2).strip()
                ub = float(v_match.group(3).strip())
                self._get_or_create_var(v_name)
                self.variables[self.var_name_to_idx[v_name]]['bounds'] = [lb, ub]
            return
            
        elif stmt.upper().startswith('@BIN'):
            v_match = re.search(r'@BIN\s*\((.*?)\)', stmt, re.IGNORECASE)
            if v_match:
                v_name = v_match.group(1).strip()
                self._get_or_create_var(v_name)
                self.variables[self.var_name_to_idx[v_name]]['type'] = 'binary'
            return
            
        elif stmt.upper().startswith('@GIN'):
            v_match = re.search(r'@GIN\s*\((.*?)\)', stmt, re.IGNORECASE)
            if v_match:
                v_name = v_match.group(1).strip()
                self._get_or_create_var(v_name)
                self.variables[self.var_name_to_idx[v_name]]['type'] = 'integer'
            return

        elif stmt.upper().startswith('@VERBOSE'):
            v_match = re.search(r'@VERBOSE\s*\((.*?)\)', stmt, re.IGNORECASE)
            if v_match:
                args = v_match.group(1).split(',')
                for arg in args:
                    if '=' in arg:
                        k, v = arg.split('=', 1)
                        self.verbose_options[k.strip().lower()] = (v.strip().lower() in ['1', 'true', 'yes', 'on'])
            return

        # 展开该语句中的所有的 @SUM / @MAX / @MIN
        stmt = self._resolve_aggregators(stmt)
        
        # 解析 目标函数 MAX / MIN
        if stmt.upper().startswith('MAX') or stmt.upper().startswith('MIN'):
            if '=' in stmt:
                obj_type, expr = stmt.split('=', 1)
            else:
                obj_type = 'min' if stmt.upper().startswith('MIN') else 'max'
                expr = stmt[3:].strip()
                
            obj_type = obj_type.strip().lower()
            self.objective['type'] = 'max' if 'max' in obj_type else 'min'
                
            if self._is_nlp(expr):
                self.objective['is_nlp'] = True
                self.objective['raw_expr'] = self._build_nlp_expr(expr)
                words = re.findall(r'[a-zA-Z_]\w*', expr)
                for w in words:
                    if w.upper() not in ['MAX', 'MIN', 'EXP', 'LOG', 'SIN', 'COS', 'TAN', 'NP']:
                        self._get_or_create_var(w)
            else:
                self.objective['expr_dict'] = self._parse_expression(expr)
            return

        # 跳过标记
        if stmt.upper() in ['SUBJECT TO', 'ST']:
            return

        # 解析普通约束
        match = re.search(r'(.*?)(<=|>=|=|<|>)(.*)', stmt)
        if match:
            lhs_str = match.group(1).strip()
            operator = match.group(2).strip()
            rhs_str = match.group(3).strip()
            
            if operator == '<': operator = '<='
            if operator == '>': operator = '>='
            
            full_expr = f"({lhs_str}) - ({rhs_str})"
            if self._is_nlp(full_expr):
                self.constraints.append({
                    'type': operator,
                    'is_nonlinear': True,
                    'raw_expr': self._build_nlp_expr(full_expr),
                    'rhs': 0.0,
                    'original': stmt
                })
                words = re.findall(r'[a-zA-Z_]\w*', full_expr)
                for w in words:
                    if w.upper() not in ['EXP', 'LOG', 'SIN', 'COS', 'TAN', 'NP']:
                        self._get_or_create_var(w)
            else:
                lhs_parsed = self._parse_expression(lhs_str)
                rhs_parsed = self._parse_expression(rhs_str)
                
                merged = {}
                for k, v in lhs_parsed.items(): merged[k] = merged.get(k, 0.0) + v
                for k, v in rhs_parsed.items(): merged[k] = merged.get(k, 0.0) - v
                
                rhs_val = -(merged.get('_constant_', 0.0))
                if '_constant_' in merged:
                    del merged['_constant_']
                    
                self.constraints.append({
                    'type': operator,
                    'expr_dict': merged,
                    'rhs': rhs_val,
                    'original': stmt
                })

    def compile(self, text: str) -> Dict[str, Any]:
        """将整段 LINGO 文本编译为标准字典。"""
        self.variables = []
        self.var_name_to_idx = {}
        self.objective = {'type': 'max', 'coeffs': []}
        self.constraints = []
        self.sets = {}
        self.data = {}
        self.verbose_options = {}
        
        # 移除注释
        text = re.sub(r'!.*?;', ';', text, flags=re.DOTALL)
        
        # 移除 LINGO 常用的无关紧要的保留字，防止用户未带分号将其与末尾变量缝合
        text = re.sub(r'(?i)\bSUBJECT\s+TO\b', '', text)
        text = re.sub(r'(?i)\bST\b', '', text)
        
        # 提取并解析区块
        text = self._parse_sets(text)
        text = self._parse_data(text)
        
        # 处理全局 @FOR
        # @FOR(CITIES(i): X(i) <= DEMAND(i));
        while True:
            blocks = self._extract_blocks(text, "@FOR")
            if not blocks:
                break
                
            start, end, outer, inner = blocks[0]
            
            parts = inner.split(':', 1)
            if len(parts) == 2:
                decl_full = parts[0].strip()
                inner_stmt = parts[1].strip()
                
                cond_expr = None
                if '|' in decl_full:
                    decl_parts = decl_full.split('|', 1)
                    iterator_decl = decl_parts[0].strip()
                    cond_expr = decl_parts[1].strip()
                else:
                    iterator_decl = decl_full
                    
                m = re.match(r'(\w+)\s*\(\s*(\w+)\s*\)', iterator_decl)
                if m:
                    set_name = m.group(1)
                    iterator = m.group(2)
                    elements = self.sets.get(set_name, {}).get('elements', [])
                    expanded_stmts = []
                    for elem in elements:
                        if cond_expr and not self._eval_condition(cond_expr, {iterator: elem}):
                            continue
                        expanded = self._expand_expression(inner_stmt, {iterator: elem})
                        expanded_stmts.append(expanded + ";")
                    replacement = " \n ".join(expanded_stmts)
                    # 替换，由于我们用分号分割，这里替换后不会影响
                    text = text[:start] + replacement + text[end:]
                    # 注意如果 FOR 后面还有分号，会在下边 strip() 掉
                    continue
            
            # 如果出错则去除这段
            text = text[:start] + " " + text[end:]
        
        # 按分号分割为语句
        statements = [stmt.strip() for stmt in text.split(';') if stmt.strip()]
        
        for stmt in statements:
            self._process_statement(stmt)

        # 映射 NLP 表达式的变量为 x[idx]
        def map_vars_to_indices(raw_str):
            # 将所有的字替换成 x[idx]
            res = raw_str
            # 按长度倒序，避免前缀被错误替换 (如 X1 包含在 X10 里)
            sorted_vars = sorted(self.variables, key=lambda v: len(v['name']), reverse=True)
            for var in sorted_vars:
                vname = var['name']
                idx = self.var_name_to_idx[vname]
                # Regex word boundary replacement
                res = re.sub(r'\b' + re.escape(vname) + r'\b', f'x[{idx}]', res)
            return res

        n_vars = len(self.variables)
        
        # 组装 objective
        if self.objective.get('is_nlp'):
            self.objective['nlp_expr'] = map_vars_to_indices(self.objective['raw_expr'])
        else:
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
            if c.get('is_nonlinear'):
                final_constraints.append({
                    'type': c['type'],
                    'is_nonlinear': True,
                    'nlp_expr': map_vars_to_indices(c['raw_expr']),
                    'rhs': c['rhs']
                })
            else:
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
            'variables': self.variables,
            'verbose_options': self.verbose_options
        }

    def compile_and_solve(self, text: str, method: str = 'auto', use_gpu: bool = None, verbose_options: dict = None) -> Dict[str, Any]:
        """编译 Lingo 风格文本并直接调用底层优化器进行求解。"""
        from ultimate_solvers_unified import quick_solve_lp
        
        compiled_data = self.compile(text)
        
        final_verbose_options = compiled_data.get('verbose_options', {})
        if verbose_options:
            final_verbose_options.update(verbose_options)
            
        result = quick_solve_lp(
            compiled_data['objective'],
            compiled_data['constraints'],
            compiled_data['variables'],
            use_gpu=use_gpu,
            method=method,
            verbose_options=final_verbose_options if final_verbose_options else None
        )
        return result

if __name__ == "__main__":
    # 测试一下高级编译器
    model_text = """
    ! 这是一个包含 SETS, DATA 和循环宏的 LINGO 模型 ;
    SETS:
        CITIES /1..3/: DEMAND, COST;
    ENDSETS

    DATA:
        DEMAND = 10 20 30;
        COST = 2 3 4;
    ENDDATA

    MAX = @SUM(CITIES(i): COST(i) * X(i));

    SUBJECT TO;
    @FOR(CITIES(i): X(i) <= DEMAND(i));
    """
    
    compiler = LingoCompiler()
    res = compiler.compile(model_text)
    print("变量表:", [v['name'] for v in res['variables']])
    print("目标函数系数:", res['objective']['coeffs'])
    print("约束:")
    for c in res['constraints']:
        print(f" {c['coeffs']} {c['type']} {c['rhs']}")
