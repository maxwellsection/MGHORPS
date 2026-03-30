import sys
import json
from lingo_compiler import LingoCompiler
import numpy as np

text = """
MAX = 2*x1 + 3*x2;
SUBJECT TO;
x1 + x2 <= 10;
x1 <= 5;
x2 <= 8;
"""
compiler = LingoCompiler()
res = compiler.compile_and_solve(text, method='builtin', verbose_options={'basic':False, 'standardize':False, 'tableau':False, 'iterations':False, 'presolve':False})
print(res.keys())
print('history length:', len(res.get('history', [])))
