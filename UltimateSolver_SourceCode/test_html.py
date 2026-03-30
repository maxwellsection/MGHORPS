import sys
import os
import json
from lingo_compiler import LingoCompiler
import numpy as np
from fractions import Fraction

with open('qt_gui/build/Desktop_Qt_6_11_0_MinGW_64_bit-Debug/temp_model.lng', 'r', encoding='utf-8') as f:
    text = f.read()

compiler = LingoCompiler()
res = compiler.compile_and_solve(text, method='builtin', verbose_options={'basic':False, 'standardize':False, 'tableau':False, 'iterations':False, 'presolve':False})
history = res.get('history', [])
print("HISTORY LENGTH:", len(history))

M_VAL = 1e6
def fmt(x):
    try:
        v = float(x)
        if abs(v) < 1e-6: return '0'
        m_coeff = round(v / M_VAL)
        rem = v - m_coeff * M_VAL
        rem_str = ''
        if abs(rem) > 1e-6:
            rem_str = str(Fraction(float(rem)).limit_denominator(100000))
        if m_coeff == 0: return rem_str
        m_str = f'{m_coeff}M' if abs(m_coeff)!=1 else ('M' if m_coeff==1 else '-M')
        if not rem_str: return m_str
        return f'{rem_str}{"+" if m_coeff>0 else ""}{m_str}'
    except: return str(x)

html = []
for idx, step in enumerate(history):
    if 'phase' in step:
        html.append(f"<h3>进入阶段 {step['phase']}</h3>")
        continue
    tab = step['tableau']
    rows, cols = tab.shape
    html.append(f"<div>Iteration {idx}</div>")
    for r in range(rows - 1):
        for j in range(cols-1):
            val = tab[r, j]
            s = fmt(val)

print("SUCCESSFULLY GENERATED HTML strings!")
