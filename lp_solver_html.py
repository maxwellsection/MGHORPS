import os
import shutil
from fractions import Fraction

class MNum:
    def __init__(self, real=0, m=0):
        if hasattr(real, "real") and hasattr(real, "m"):
            self.real = Fraction(real.real)
            self.m = Fraction(real.m)
        else:
            self.real = Fraction(real)
            self.m = Fraction(m)

    def __add__(self, other):
        other = MNum(other)
        return MNum(self.real + other.real, self.m + other.m)

    def __sub__(self, other):
        other = MNum(other)
        return MNum(self.real - other.real, self.m - other.m)

    def __neg__(self):
        return MNum(-self.real, -self.m)

    def __mul__(self, other):
        if isinstance(other, MNum):
            if self.m != 0 and other.m != 0: raise ValueError("M*M")
            return MNum(self.real * other.real, self.real * other.m + self.m * other.real)
        return MNum(self.real * Fraction(other), self.m * Fraction(other))

    def __truediv__(self, other):
        r = Fraction(getattr(other, "real", other))
        return MNum(self.real / r, self.m / r)

    def __lt__(self, other):
        other = MNum(other)
        if self.m < other.m: return True
        if self.m > other.m: return False
        return self.real < other.real
        
    def __le__(self, other): return self < other or self == other
    def __eq__(self, other):
        other = MNum(other)
        return self.real == other.real and self.m == other.m
    def __gt__(self, other): return not (self <= other)
    def __ge__(self, other): return not (self < other)

    def __str__(self):
        def f(num): return str(num.numerator) if num.denominator == 1 else f"{num.numerator}/{num.denominator}"
        if self.m == 0: return f(self.real)
        if self.real == 0:
            if self.m == 1: return "M"
            if self.m == -1: return "-M"
            return f"{f(self.m)}M"
        if self.m > 0:
            m_str = "M" if self.m == 1 else f"{f(self.m)}M"
            return f"{f(self.real)}+{m_str}"
        m_str = "-M" if self.m == -1 else f"{f(self.m)}M"
        return f"{f(self.real)}{m_str}"

HTML_HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f6fa; color: #333; margin: 40px; }
h1, h2, h3 { color: #2c3e50; }
.problem-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 40px; }
.iteration { margin-top: 20px; margin-bottom: 20px; }
table { border-collapse: collapse; text-align: center; font-size: 16px; margin-bottom: 10px; }
th, td { border: 1px solid #ccc; padding: 8px 12px; }
.top-row { border-bottom: 2px solid #333; }
.header-row { border-bottom: 2px solid #333; background: #f0f2f5; font-weight: bold; }
.c-col { border-right: 2px solid #333; }
.bottom-row { border-top: 2px solid #333; font-weight: bold; }
.info { color: #e74c3c; font-weight: bold; margin-bottom: 15px; }
.highlight { outline: 3px solid #3498db; outline-offset: -2px; }
.optimal { border: 2px solid #2ecc71; padding: 10px; background: #eaeff2; border-radius: 5px; }
</style>
</head>
<body>
<h1>运筹学单纯形表 (可视化演示)</h1>
"""

HTML_FOOT = """</body></html>"""

class SimplexProblem:
    def __init__(self, c_real, c_m, A, b, basis, varnames, is_max, problem_name, original_is_min):
        self.c_real = c_real
        self.c_m = c_m
        self.A = A
        self.b = b
        self.basis = list(basis)
        self.varnames = varnames
        self.is_max = is_max
        self.original_is_min = original_is_min
        self.problem_name = problem_name
        self.obj = [MNum(cr, cm) for cr, cm in zip(c_real, c_m)]

    def _render_table_html(self, f, iteration, m, n, obj, tab, basis, varnames, c_minus_z, z_val, pivot_pos, z_label="-z", is_phase_1=False):
        f.write(f"<div class='iteration'><strong>第 {iteration} 步迭代:</strong><br><br>")
        f.write("<table>\n")
        # Top C row
        f.write("<tr class='top-row'>")
        f.write(f"<td colspan='3' class='c-col'>C</td>")
        for j in range(n): f.write(f"<td>{obj[j]}</td>")
        f.write("</tr>\n")
        
        # Header row
        f.write("<tr class='header-row'>")
        f.write("<td>C<sub>B</sub></td><td>X<sub>B</sub></td><td class='c-col'>b</td>")
        for v in varnames: f.write(f"<td>{v}</td>")
        f.write("</tr>\n")
        
        # Body
        for i in range(m):
            f.write("<tr>")
            f.write(f"<td>{obj[basis[i]]}</td><td>{varnames[basis[i]]}</td><td class='c-col'>{tab[i][-1]}</td>")
            for j in range(n):
                cls = "highlight" if pivot_pos and pivot_pos[0] == i and pivot_pos[1] == j else ""
                f.write(f"<td class='{cls}'>{tab[i][j]}</td>")
            f.write("</tr>\n")
            
        # Bottom row (-z = ...)
        f.write("<tr class='bottom-row'>")
        # In the image, bottom row spans the left 3 columns for '-z=-val'
        mz_v = -z_val if self.is_max else z_val
        # Wait, if `z_val` is the actual z from the tableau (sum cB * xB), then in the tableau we track the value of the objective.
        # Actually our problem objective is `z'`. If original is min, z' = -z. 
        # Tableau calculates `z'`. The value is `z_val`. 
        # The equation is `z' = z_val`. Thus `-z = z_val` if original is min.
        # So we just print `z_label = z_val`.
        f.write(f"<td colspan='3' class='c-col'>{z_label}={mz_v}</td>")
        
        for j in range(n):
            f.write(f"<td>{c_minus_z[j]}</td>")
        f.write("</tr>\n")
        
        f.write("</table></div>\n")

    def run_big_m(self, f):
        f.write(f"<div class='problem-card'><h2>{self.problem_name} - 大M法 (Big M Method)</h2>\n")
        m, n = len(self.A), len(self.A[0])
        tab = [[MNum(self.A[i][j]) for j in range(n)] + [MNum(self.b[i])] for i in range(m)]
        basis = list(self.basis)
        
        z_label = "-z" if self.original_is_min else "z"
        
        loop = 1
        while True:
            c_minus_z = []
            for j in range(n):
                zj = sum((self.obj[basis[i]] * tab[i][j] for i in range(m)), MNum(0))
                c_minus_z.append(self.obj[j] - zj)
            
            z_val = sum((self.obj[basis[i]] * tab[i][-1] for i in range(m)), MNum(0))
            
            if all(x <= MNum(0) for x in c_minus_z):
                self._render_table_html(f, loop, m, n, self.obj, tab, basis, self.varnames, c_minus_z, z_val, None, z_label)
                has_artificial = any(self.varnames[basis[i]].startswith('a') and tab[i][-1] > MNum(0) for i in range(m))
                if has_artificial:
                    f.write("<div class='info'>⚠️ 人工变量仍然留在基变量中且不为零，因此原问题不可行 (Infeasible)！</div>\n")
                else:
                    opt_z = -z_val if self.original_is_min else z_val
                    f_obj_name = "min z" if self.original_is_min else "max z"
                    f.write(f"<div class='optimal'>✅ 已达最优解！原问题最优值 <strong>{f_obj_name} = {opt_z}</strong></div>\n")
                break
                
            enter = max(range(n), key=lambda j: c_minus_z[j])
            if c_minus_z[enter] <= MNum(0): break
            
            leave, best_ratio_val = -1, float('inf')
            for i in range(m):
                if tab[i][enter] > MNum(0):
                    rv = tab[i][-1] / tab[i][enter]
                    if rv.real < best_ratio_val:
                        best_ratio_val, leave = float(rv.real), i
            
            if leave == -1:
                self._render_table_html(f, loop, m, n, self.obj, tab, basis, self.varnames, c_minus_z, z_val, None, z_label)
                f.write("<div class='info'>⚠️ 所有选定列的约束系数 <= 0，因此问题无界 (Unbounded)！</div>\n")
                break
                
            self._render_table_html(f, loop, m, n, self.obj, tab, basis, self.varnames, c_minus_z, z_val, (leave, enter), z_label)
            f.write(f"<div class='info'>🔄 换基: 进基变量为 <strong>{self.varnames[enter]}</strong>，出基变量为 <strong>{self.varnames[basis[leave]]}</strong></div>\n")
            
            pivot = tab[leave][enter]
            for j in range(n+1): tab[leave][j] = tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = tab[i][enter]
                    for j in range(n+1): tab[i][j] = tab[i][j] - factor * tab[leave][j]
            basis[leave] = enter
            loop += 1
        f.write("</div>\n")

    def run_two_phase(self, f):
        f.write(f"<div class='problem-card'><h2>{self.problem_name} - 两阶段法 (Two Phase Method)</h2>\n")
        art_idx = [i for i, n in enumerate(self.varnames) if n.startswith('a')]
        if not art_idx:
            f.write("<div class='info'>此问题无需人工变量，无需使用两阶段法拆分计算。</div></div>\n")
            return
            
        f.write("<h3>第一阶段 (Phase I)</h3>\n")
        m, n = len(self.A), len(self.A[0])
        tab = [[MNum(self.A[i][j]) for j in range(n)] + [MNum(self.b[i])] for i in range(m)]
        basis = list(self.basis)
        obj1 = [MNum(-1) if i in art_idx else MNum(0) for i in range(n)]
        
        loop = 1
        while True:
            c_minus_z = []
            for j in range(n):
                zj = sum((obj1[basis[i]] * tab[i][j] for i in range(m)), MNum(0))
                c_minus_z.append(obj1[j] - zj)
            w_val = sum((obj1[basis[i]] * tab[i][-1] for i in range(m)), MNum(0))
            
            if all(x <= MNum(0) for x in c_minus_z):
                self._render_table_html(f, loop, m, n, obj1, tab, basis, self.varnames, c_minus_z, w_val, None, "-w", True)
                f.write("<div class='optimal'>👉 第一阶段结束。</div>\n")
                break
                
            enter = max(range(n), key=lambda j: c_minus_z[j])
            if c_minus_z[enter] <= MNum(0): break
            leave, best_ratio_val = -1, float('inf')
            for i in range(m):
                if tab[i][enter] > MNum(0):
                    rv = tab[i][-1] / tab[i][enter]
                    if rv.real < best_ratio_val:
                        best_ratio_val, leave = float(rv.real), i
                        
            if leave == -1:
                self._render_table_html(f, loop, m, n, obj1, tab, basis, self.varnames, c_minus_z, w_val, None, "-w", True)
                f.write("<div class='info'>⚠️ 第一阶段无界。</div>\n")
                f.write("</div>")
                return
                
            self._render_table_html(f, loop, m, n, obj1, tab, basis, self.varnames, c_minus_z, w_val, (leave, enter), "-w", True)
            f.write(f"<div class='info'>🔄 换基: 进基 <strong>{self.varnames[enter]}</strong>，出基 <strong>{self.varnames[basis[leave]]}</strong></div>\n")
            
            pivot = tab[leave][enter]
            for j in range(n+1): tab[leave][j] = tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = tab[i][enter]
                    for j in range(n+1): tab[i][j] = tab[i][j] - factor * tab[leave][j]
            basis[leave] = enter
            loop += 1
            
        if w_val.real < 0:
            f.write("<div class='info'>⚠️ 第一阶段最优值 -w < 0，因此原问题不可行 (Infeasible)！</div></div>\n")
            return
            
        f.write("<h3>第二阶段 (Phase II)</h3>\n")
        new_varnames = [v for i, v in enumerate(self.varnames) if i not in art_idx]
        new_obj = [self.obj[i] for i in range(n) if i not in art_idx]
        old2new = {i: idx for idx, i in enumerate(i for i in range(n) if i not in art_idx)}
        new_basis = []
        for b in basis:
            if b in old2new: new_basis.append(old2new[b])
            else:
                f.write("<div class='info'>⚠️ 退化情况导致人工变量仍在基中，跳过此阶段。</div></div>\n")
                return
                
        new_n = len(new_varnames)
        new_tab = [[tab[i][j] for j in range(n) if j not in art_idx] + [tab[i][-1]] for i in range(m)]
        z_label = "-z" if self.original_is_min else "z"
        
        loop = 1
        while True:
            c_minus_z = []
            for j in range(new_n):
                zj = sum((new_obj[new_basis[i]] * new_tab[i][j] for i in range(m)), MNum(0))
                c_minus_z.append(new_obj[j] - zj)
            z_val = sum((new_obj[new_basis[i]] * new_tab[i][-1] for i in range(m)), MNum(0))
            
            if all(x <= MNum(0) for x in c_minus_z):
                self._render_table_html(f, loop, m, new_n, new_obj, new_tab, new_basis, new_varnames, c_minus_z, z_val, None, z_label)
                opt_z = -z_val if self.original_is_min else z_val
                f_obj_name = "min z" if self.original_is_min else "max z"
                f.write(f"<div class='optimal'>✅ 已达最优解！原问题最优值 <strong>{f_obj_name} = {opt_z}</strong></div>\n")
                break
                
            enter = max(range(new_n), key=lambda j: c_minus_z[j])
            if c_minus_z[enter] <= MNum(0): break
            leave, best_ratio_val = -1, float('inf')
            for i in range(m):
                if new_tab[i][enter] > MNum(0):
                    rv = new_tab[i][-1] / new_tab[i][enter]
                    if rv.real < best_ratio_val:
                        best_ratio_val, leave = float(rv.real), i
                        
            if leave == -1:
                self._render_table_html(f, loop, m, new_n, new_obj, new_tab, new_basis, new_varnames, c_minus_z, z_val, None, z_label)
                f.write("<div class='info'>⚠️ 无界 (Unbounded)！</div>\n")
                break
                
            self._render_table_html(f, loop, m, new_n, new_obj, new_tab, new_basis, new_varnames, c_minus_z, z_val, (leave, enter), z_label)
            f.write(f"<div class='info'>🔄 换基: 进基 <strong>{new_varnames[enter]}</strong>，出基 <strong>{new_varnames[new_basis[leave]]}</strong></div>\n")
            pivot = new_tab[leave][enter]
            for j in range(new_n+1): new_tab[leave][j] = new_tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = new_tab[i][enter]
                    for j in range(new_n+1): new_tab[i][j] = new_tab[i][j] - factor * new_tab[leave][j]
            new_basis[leave] = enter
            loop += 1
            
        f.write("</div>\n")

out_path = r"..\运筹学作业最终版\单纯形表可视化.html"

def build_all():
    probs = []
    # original_is_min helps set the sign for Final Z and the label in the tableau perfectly.
    probs.append(SimplexProblem(
        c_real=[-5, -1, 0, 0, 0], c_m=[0, 0, 0, 0, -1],
        A=[[-1, 1, 1, 0, 0], [1, 1, 0, -1, 1]], b=[1, 2],
        basis=[2, 4], varnames=['x1', 'x2', 's1', 's2', 'a1'], is_max=True, problem_name="题目 1 (原问题 min z)", original_is_min=True
    ))
    probs.append(SimplexProblem(
        c_real=[-2, -3, -1, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, -1, -1],
        A=[[1, 4, 2, -1, 0, 1, 0], [3, 2, 0, 0, -1, 0, 1]], b=[8, 6],
        basis=[5, 6], varnames=['x1', 'x2', 'x3', 's1', 's2', 'a1', 'a2'], is_max=True, problem_name="题目 2 (原问题 min z)", original_is_min=True
    ))
    probs.append(SimplexProblem(
        c_real=[-4, -1, 0, 0, 0, 0], c_m=[0, 0, 0, 0, -1, -1],
        A=[[3, 1, 0, 0, 1, 0], [4, 3, -1, 0, 0, 1], [1, 2, 0, 1, 0, 0]], b=[3, 6, 4],
        basis=[4, 5, 3], varnames=['x1', 'x2', 'x3', 'x4', 'a1', 'a2'], is_max=True, problem_name="题目 3 (原问题 min z)", original_is_min=True
    ))
    probs.append(SimplexProblem(
        c_real=[2, -1, 2, 0, 0, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, 0, -1, -1, -1],
        A=[[1, 1, 1, -1, 0, 0, 1, 0, 0], [-2, 0, 1, 0, -1, 0, 0, 1, 0], [0, 2, -1, 0, 0, -1, 0, 0, 1]], b=[6, 2, 0],
        basis=[6, 7, 8], varnames=['x1', 'x2', 'x3', 's1', 's2', 's3', 'a1', 'a2', 'a3'], is_max=True, problem_name="题目 4 (原问题 max z)", original_is_min=False
    ))
    probs.append(SimplexProblem(
        c_real=[3, 2, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, -1],
        A=[[1, 1, 1, 0, 0, 0], [-2, 3, 0, 1, 0, 0], [1, 0, 0, 0, -1, 1]], b=[4, 6, 5],
        basis=[2, 3, 5], varnames=['x1', 'x2', 's1', 's2', 's3', 'a1'], is_max=True, problem_name="题目 5 (原问题 max z)", original_is_min=False
    ))
    probs.append(SimplexProblem(
        c_real=[-1, 2, -3, 0, 0, 0, 0, 0], c_m=[0]*6 + [-1, -1],
        A=[[1, 1, 1, -1, 0, 0, 1, 0], [2, 0, 1, 0, 1, 0, 0, 0], [0, 1, 2, 0, 0, -1, 0, 1]], b=[6, 8, 10],
        basis=[6, 4, 7], varnames=['x1', 'x2', 'x3', 's1', 's2', 's3', 'a1', 'a2'], is_max=True, problem_name="题目 6 (原问题 min z)", original_is_min=True
    ))
    probs.append(SimplexProblem(
        c_real=[2, 4, 1, 2, 0, 0, 0], c_m=[0]*6 + [-1],
        A=[[2, 1, -1, 0, -1, 0, 1], [1, 2, 3, 0, 0, 1, 0], [1, 4, 2, 1, 0, 0, 0]], b=[20, 30, 40],
        basis=[6, 5, 3], varnames=['x1', 'x2', 'x3', 'x4', 's1', 's2', 'a1'], is_max=True, problem_name="题目 7 (原问题 max z)", original_is_min=False
    ))
    probs.append(SimplexProblem(
        c_real=[-4, 2, 0, 0, 0, 0, 0, 0], c_m=[0]*6 + [-1, -1],
        A=[[-3, 4, 1, 0, 0, 0, 0, 0], [6, -3, 0, 1, 0, 0, 0, 0], [2, 1, 0, 0, -1, 0, 1, 0], [1, 1, 0, 0, 0, -1, 0, 1]], b=[36, 24, 4, 10],
        basis=[2, 3, 6, 7], varnames=['x1', 'x2', 's1', 's2', 's3', 's4', 'a1', 'a2'], is_max=True, problem_name="题目 8 (原问题 min z)", original_is_min=True
    ))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEAD)
        for p in probs:
            p.run_big_m(f)
            p.run_two_phase(f)
        f.write(HTML_FOOT)

import re

def negate_expr(expr):
    expr = expr.strip()
    if not expr: return expr
    expr = expr.replace(' ', '')
    if not expr.startswith('-') and not expr.startswith('+'):
        expr = '+' + expr
    terms = re.findall(r'[+-][^+-]+', expr)
    new_terms = []
    for t in terms:
        if t.startswith('+'): new_terms.append('-' + t[1:])
        else: new_terms.append('+' + t[1:])
    res = ''.join(new_terms)
    if res.startswith('+'): res = res[1:]
    if res == '-0': res = '0'
    return res

def process_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    out_lines = []
    current_mapping = {}
    
    for i, line in enumerate(lines):
        if '<h2>题目' in line and '大M法' in line:
            current_mapping = {}
            
        if "<tr class='header-row'>" in line:
            tds = re.findall(r'<td>(.*?)</td>', line)
            if not tds:
                tds = re.findall(r'<td[^>]*>(.*?)</td>', line)
            max_x = 0
            for c in tds:
                clean_c = re.sub(r'<[^>]+>', '', c).strip()
                m = re.search(r'x(\d+)', clean_c)
                if m: max_x = max(max_x, int(m.group(1)))
            
            next_x = max_x + 1
            for c in tds:
                clean_c = re.sub(r'<[^>]+>', '', c).strip()
                if clean_c.startswith('s') or clean_c.startswith('a') or clean_c.startswith('x'):
                    if clean_c.startswith('s') or clean_c.startswith('a'):
                        if clean_c not in current_mapping and re.match(r'^[as]\d+$', clean_c):
                            current_mapping[clean_c] = f"x{next_x}"
                            next_x += 1

        if "<tr class='bottom-row'>" in line:
            def repl_obj(m):
                prefix = m.group(1)
                obj_name = m.group(2)
                val = m.group(3)
                new_val = negate_expr(val)
                return f"{prefix}{obj_name}{new_val}</td>"
            line = re.sub(r'(<td[^>]*>)(-z=|z=|-w=)(.*?)</td>', repl_obj, line)
            
        new_line = line
        for old_v, new_v in current_mapping.items():
            new_line = re.sub(r'(?<![a-zA-Z0-9])' + old_v + r'(?![a-zA-Z0-9])', new_v, new_line)
        out_lines.append(new_line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))


if __name__ == "__main__":
    build_all()
    process_html_file(out_path)
    print("HTML Generator complete and fixed.")
