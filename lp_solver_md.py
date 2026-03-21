import os
import shutil
from fractions import Fraction
import copy

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

def format_row(row):
    return " | ".join(str(x) for x in row)

class SimplexProblem:
    def __init__(self, c_real, c_m, A, b, basis, varnames, is_max, problem_name):
        self.c_real = c_real
        self.c_m = c_m
        self.A = A
        self.b = b
        self.basis = list(basis)
        self.varnames = varnames
        self.is_max = is_max
        self.problem_name = problem_name
        self.obj = [MNum(cr, cm) for cr, cm in zip(c_real, c_m)]

    def run_big_m(self, f):
        f.write(f"\n### {self.problem_name} - 大M法 (Big M Method)\n\n")
        m, n = len(self.A), len(self.A[0])
        tab = [[MNum(self.A[i][j]) for j in range(n)] + [MNum(self.b[i])] for i in range(m)]
        basis = list(self.basis)
        
        loop = 1
        while True:
            f.write(f"**第 {loop} 步迭代:**\n\n")
            f.write("| $C_B$ | $X_B$ | $b$ | " + " | ".join(f"${v}$" for v in self.varnames) + " |\n")
            f.write("|---|---|---|" + "|".join(["---"]*n) + "|\n")
            c_minus_z = []
            for j in range(n):
                zj = MNum(0)
                for i in range(m): zj += self.obj[basis[i]] * tab[i][j]
                c_minus_z.append(self.obj[j] - zj)
            
            z_val = MNum(0)
            for i in range(m): z_val += self.obj[basis[i]] * tab[i][-1]
            
            for i in range(m):
                cb = str(self.obj[basis[i]])
                xb = f"${self.varnames[basis[i]]}$"
                r_vals = [str(x) for x in tab[i]]
                f.write(f"| {cb} | {xb} | {r_vals[-1]} | {format_row(r_vals[:-1])} |\n")
                
            mz = "-z" if self.is_max else "z"
            mz_v = -z_val if self.is_max else z_val
            f.write(f"| | ${mz}$ | {mz_v} | {format_row(c_minus_z)} |\n")
            
            if all(x <= MNum(0) for x in c_minus_z):
                f.write("\\n👉 检测到所有检验数均非正 $\\sigma_j \\le 0$，当前即为最优解/最终表！\\n")
                has_artificial = any(self.varnames[basis[i]].startswith('a') and tab[i][-1] > MNum(0) for i in range(m))
                if has_artificial:
                    f.write("\n⚠️ 人工变量仍然留在基变量中且不为零，因此原问题**不可行 (Infeasible)**！\n")
                else:
                    opt_z = z_val if self.is_max else -z_val
                    f.write(f"\n✅ 最优目标值 Z = {opt_z}\n\n")
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
                f.write("\n⚠️ 所有选定列的约束系数 <= 0，因此问题**无界 (Unbounded)**！\n\n")
                break
                
            f.write(f"\n🔄 **换基:** 进基变量为 `${self.varnames[enter]}$`，出基变量为 `${self.varnames[basis[leave]]}$`。主元: {tab[leave][enter]}\n\n")
            
            pivot = tab[leave][enter]
            for j in range(n+1): tab[leave][j] = tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = tab[i][enter]
                    for j in range(n+1): tab[i][j] = tab[i][j] - factor * tab[leave][j]
            basis[leave] = enter
            loop += 1

    def run_two_phase(self, f):
        f.write(f"\n---\n### {self.problem_name} - 两阶段法 (Two Phase Method)\n\n")
        art_idx = [i for i, n in enumerate(self.varnames) if n.startswith('a')]
        if not art_idx:
            f.write("此问题无需人工变量，无需使用两阶段法拆分计算。\n")
            return
            
        f.write("#### 第一阶段 (Phase I)\n\n")
        f.write(f"目标函数变为: $\\max(-w) = " + " - ".join(self.varnames[i] for i in art_idx) + "$\n\n")
        m, n = len(self.A), len(self.A[0])
        tab = [[MNum(self.A[i][j]) for j in range(n)] + [MNum(self.b[i])] for i in range(m)]
        basis = list(self.basis)
        obj1 = [MNum(-1) if i in art_idx else MNum(0) for i in range(n)]
        
        loop = 1
        while True:
            f.write(f"**第 {loop} 步 (第一阶段):**\n\n")
            f.write("| CB | XB | b | " + " | ".join(f"${v}$" for v in self.varnames) + " |\n")
            f.write("|---|---|---|" + "|".join(["---"]*n) + "|\n")
            c_minus_z = []
            for j in range(n):
                zj = sum((obj1[basis[i]] * tab[i][j] for i in range(m)), MNum(0))
                c_minus_z.append(obj1[j] - zj)
            w_val = sum((obj1[basis[i]] * tab[i][-1] for i in range(m)), MNum(0))
            
            for i in range(m):
                f.write(f"| {obj1[basis[i]]} | ${self.varnames[basis[i]]}$ | {tab[i][-1]} | {format_row(tab[i][:-1])} |\n")
            f.write(f"| | -W | {-w_val} | {format_row(c_minus_z)} |\n")
            
            if all(x <= MNum(0) for x in c_minus_z):
                f.write("\\n👉 $\\sigma_j \\le 0$，Phase I 结束。\\n")
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
                f.write("\n⚠️ 第一阶段无界，问题无界。\n")
                return
            f.write(f"\n🔄 **换基:** 进基 `${self.varnames[enter]}$`，出基 `${self.varnames[basis[leave]]}$`。主元: {tab[leave][enter]}\n\n")
            pivot = tab[leave][enter]
            for j in range(n+1): tab[leave][j] = tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = tab[i][enter]
                    for j in range(n+1): tab[i][j] = tab[i][j] - factor * tab[leave][j]
            basis[leave] = enter
            loop += 1
            
        if w_val.real < 0:
            f.write("\n⚠️ 结束时 $-w < 0$，因此原问题**不可行 (Infeasible)**！直接结束求解。\n\n")
            return
            
        f.write("\n#### 第二阶段 (Phase II)\n\n")
        f.write("去除人工变量，恢复原目标函数继续迭代。\n\n")
        new_varnames = [v for i, v in enumerate(self.varnames) if i not in art_idx]
        new_obj = [self.obj[i] for i in range(n) if i not in art_idx]
        
        # Artificial variables must be out of basis. If they are in basis with value 0, drop them or just skip
        old2new = {i: idx for idx, i in enumerate(i for i in range(n) if i not in art_idx)}
        new_basis = []
        for b in basis:
            if b in old2new: new_basis.append(old2new[b])
            else:
                f.write("\n⚠️ 退化情况导致人工变量仍在基中，简化处理将跳过该问题继续。\n")
                return
                
        new_n = len(new_varnames)
        new_tab = [[tab[i][j] for j in range(n) if j not in art_idx] + [tab[i][-1]] for i in range(m)]
        
        loop = 1
        while True:
            f.write(f"**第 {loop} 步 (第二阶段):**\n\n")
            f.write("| $C_B$ | $X_B$ | $b$ | " + " | ".join(f"${v}$" for v in new_varnames) + " |\n")
            f.write("|---|---|---|" + "|".join(["---"]*new_n) + "|\n")
            c_minus_z = []
            for j in range(new_n):
                zj = sum((new_obj[new_basis[i]] * new_tab[i][j] for i in range(m)), MNum(0))
                c_minus_z.append(new_obj[j] - zj)
            z_val = sum((new_obj[new_basis[i]] * new_tab[i][-1] for i in range(m)), MNum(0))
            for i in range(m):
                f.write(f"| {new_obj[new_basis[i]]} | ${new_varnames[new_basis[i]]}$ | {new_tab[i][-1]} | {format_row(new_tab[i][:-1])} |\n")
            mz = "-z" if self.is_max else "z"
            mz_v = -z_val if self.is_max else z_val
            f.write(f"| | ${mz}$ | {mz_v} | {format_row(c_minus_z)} |\n")
            if all(x <= MNum(0) for x in c_minus_z):
                f.write("\\n👉 检测到所有检验数均非正 $\\sigma_j \\le 0$，已达到最优解！\\n")
                opt_z = z_val if self.is_max else -z_val
                f.write(f"\n✅ 最优目标值 Z = {opt_z}\n\n")
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
                f.write("\n⚠️ **无界 (Unbounded)**！\n\n")
                break
            f.write(f"\n🔄 **换基:** 进基 `${new_varnames[enter]}$`，出基 `${new_varnames[new_basis[leave]]}$`。主元: {new_tab[leave][enter]}\n\n")
            pivot = new_tab[leave][enter]
            for j in range(new_n+1): new_tab[leave][j] = new_tab[leave][j] / pivot
            for i in range(m):
                if i != leave:
                    factor = new_tab[i][enter]
                    for j in range(new_n+1): new_tab[i][j] = new_tab[i][j] - factor * new_tab[leave][j]
            new_basis[leave] = enter
            loop += 1

out_path = r"..\运筹学作业最终版\各题大M法与两阶段法解题过程.md"

def build_all():
    probs = []
    probs.append(SimplexProblem(
        c_real=[-5, -1, 0, 0, 0], c_m=[0, 0, 0, 0, -1],
        A=[[-1, 1, 1, 0, 0], [1, 1, 0, -1, 1]], b=[1, 2],
        basis=[2, 4], varnames=['x_1', 'x_2', 's_1', 's_2', 'a_1'], is_max=True, problem_name="题目 1"
    ))
    probs.append(SimplexProblem(
        c_real=[-2, -3, -1, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, -1, -1],
        A=[[1, 4, 2, -1, 0, 1, 0], [3, 2, 0, 0, -1, 0, 1]], b=[8, 6],
        basis=[5, 6], varnames=['x_1', 'x_2', 'x_3', 's_1', 's_2', 'a_1', 'a_2'], is_max=True, problem_name="题目 2"
    ))
    probs.append(SimplexProblem(
        c_real=[-4, -1, 0, 0, 0, 0], c_m=[0, 0, 0, 0, -1, -1],
        A=[[3, 1, 0, 0, 1, 0], [4, 3, -1, 0, 0, 1], [1, 2, 0, 1, 0, 0]], b=[3, 6, 4],
        basis=[4, 5, 3], varnames=['x_1', 'x_2', 'x_3', 'x_4', 'a_1', 'a_2'], is_max=True, problem_name="题目 3"
    ))
    probs.append(SimplexProblem(
        c_real=[2, -1, 2, 0, 0, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, 0, -1, -1, -1],
        A=[[1, 1, 1, -1, 0, 0, 1, 0, 0], [-2, 0, 1, 0, -1, 0, 0, 1, 0], [0, 2, -1, 0, 0, -1, 0, 0, 1]], b=[6, 2, 0],
        basis=[6, 7, 8], varnames=['x_1', 'x_2', 'x_3', 's_1', 's_2', 's_3', 'a_1', 'a_2', 'a_3'], is_max=True, problem_name="题目 4"
    ))
    probs.append(SimplexProblem(
        c_real=[3, 2, 0, 0, 0, 0], c_m=[0, 0, 0, 0, 0, -1],
        A=[[1, 1, 1, 0, 0, 0], [-2, 3, 0, 1, 0, 0], [1, 0, 0, 0, -1, 1]], b=[4, 6, 5],
        basis=[2, 3, 5], varnames=['x_1', 'x_2', 's_1', 's_2', 's_3', 'a_1'], is_max=True, problem_name="题目 5"
    ))
    probs.append(SimplexProblem(
        c_real=[-1, 2, -3, 0, 0, 0, 0, 0], c_m=[0]*6 + [-1, -1],
        A=[[1, 1, 1, -1, 0, 0, 1, 0], [2, 0, 1, 0, 1, 0, 0, 0], [0, 1, 2, 0, 0, -1, 0, 1]], b=[6, 8, 10],
        basis=[6, 4, 7], varnames=['x_1', 'x_2', 'x_3', 's_1', 's_2', 's_3', 'a_1', 'a_2'], is_max=True, problem_name="题目 6"
    ))
    probs.append(SimplexProblem(
        c_real=[2, 4, 1, 2, 0, 0, 0], c_m=[0]*6 + [-1],
        A=[[2, 1, -1, 0, -1, 0, 1], [1, 2, 3, 0, 0, 1, 0], [1, 4, 2, 1, 0, 0, 0]], b=[20, 30, 40],
        basis=[6, 5, 3], varnames=['x_1', 'x_2', 'x_3', 'x_4', 's_1', 's_2', 'a_1'], is_max=True, problem_name="题目 7"
    ))
    probs.append(SimplexProblem(
        c_real=[-4, 2, 0, 0, 0, 0, 0, 0], c_m=[0]*6 + [-1, -1],
        A=[[-3, 4, 1, 0, 0, 0, 0, 0], [6, -3, 0, 1, 0, 0, 0, 0], [2, 1, 0, 0, -1, 0, 1, 0], [1, 1, 0, 0, 0, -1, 0, 1]], b=[36, 24, 4, 10],
        basis=[2, 3, 6, 7], varnames=['x_1', 'x_2', 's_1', 's_2', 's_3', 's_4', 'a_1', 'a_2'], is_max=True, problem_name="题目 8"
    ))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 运筹学单纯形表 (大M法与两阶段法) 解题过程\n\n")
        f.write("> 本文档包含所有8道题目的严谨演算过程，完全对应单纯形表的标准格式要求。\n\n")
        
        for p in probs:
            p.run_big_m(f)
            p.run_two_phase(f)
            f.write("\n\n---\n\n")

if __name__ == "__main__":
    build_all()
    # Move files
    files_to_move = ['solve_all.py', 'fixer.py', 'output.txt', 'output_utf8.txt']
    for f in files_to_move:
        if os.path.exists(f):
            shutil.move(f, f"..\\运筹学作业最终版\\{f}")
