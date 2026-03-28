import time
import random
from milp_solver import MILPBranchAndBound

# 生成一个复杂的多维背包模型
# 物品数量
N_ITEMS = 30
# 维度数量 (如果是1则为简单背包，大于1则是多维背包)
N_DIMS = 3

random.seed(42)

# 目标：最大化价值
objs = [random.randint(10, 100) for _ in range(N_ITEMS)]
objective = {'type': 'maximize', 'coeffs': objs}

constraints = []
# 生成多维重量
for dim in range(N_DIMS):
    weights = [random.randint(5, 50) for _ in range(N_ITEMS)]
    # 容量大约让三分之一的物品能装入
    capacity = sum(weights) / 3.0
    constraints.append({
        'type': '<=',
        'coeffs': weights,
        'rhs': capacity
    })

# 所有都是 0-1 变量
variables = []
for i in range(N_ITEMS):
    variables.append({'name': f'x{i}', 'type': 'binary'})

print(f"=== 运行多维背包问题 (N={N_ITEMS}, Dims={N_DIMS}) ===")
# 使用 Pseudo-cost 策略的内置求解器
solver = MILPBranchAndBound(solver='builtin', time_limit=120)

t0 = time.time()
res = solver.solve(objective, constraints, variables)
t1 = time.time()

print(f"\\n最终结果:")
print(f"探索节点: {res.get('nodes_explored')}")
print(f"目标值: {res.get('objective_value')}")
print(f"耗时: {t1-t0:.2f}s")
