from lingo_compiler import LingoCompiler
import json

model_text = """
SETS:
    CITIES /1..3/: DEMAND, COST;
ENDSETS

DATA:
    DEMAND = 10 20 30;
    COST = 2 3 4;
ENDDATA

MAX = @SUM(CITIES(i) | i #EQ# 1: COST(i) * X(i)) + @MAX(CITIES(i): COST(i)) * Y;

@FOR(CITIES(i) | i #GT# 1: X(i) <= DEMAND(i));

@BIN(Y);
@GIN(X_2);
"""

compiler = LingoCompiler()
res = compiler.compile(model_text)

print(json.dumps({
    'variables': res['variables'],
    'objective': res['objective'],
    'constraints': res['constraints']
}, indent=2, ensure_ascii=False))
