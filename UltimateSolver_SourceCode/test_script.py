import sys
import json
from lingo_compiler import LingoCompiler
import numpy as np

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    text = f.read()

compiler = LingoCompiler()
res = compiler.compile_and_solve(text, method='builtin', verbose_options={'basic':True, 'standardize':True, 'tableau':True, 'iterations':True, 'presolve':False})
print("STATUS:", res.get('status'))
print("LEN HISTORY:", len(res.get('history', [])))
