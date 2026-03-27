import ultimate_opt as opt

print("=== 模式1: PyTorch 面向对象风格 ===")

class FactoryModel(opt.Model):
    def __init__(self):
        super().__init__()
        # 像定义 PyTorch 网络层 / Parameter 一样定义决策变量
        self.ProductA = opt.Variable(name="ProductA", lower_bound=0)
        self.ProductB = opt.Variable(name="ProductB", lower_bound=0)

    def forward(self):
        # 像编写 Lingo 公式一样声明约束
        self.subject_to(
            self.ProductA + self.ProductB <= 50,
            3 * self.ProductA + 2 * self.ProductB <= 100,
            # 可以直接处理更复杂的表达式和负数、减法
            self.ProductA - 0.5 * self.ProductB >= 0
        )
        # 声明目标函数
        return self.maximize(20 * self.ProductA + 30 * self.ProductB)

# 实例化并构建模型
model = FactoryModel()

# 实例化引擎并求解
solver = opt.Solver(method='builtin')
print("正在求解...")
result = solver.solve(model)
print(f"最优化结果状态: {result.get('status')}")
print(f"目标函数值: {result.get('objective_value')}")
print(f"最优解: {result.get('solution')}")
print("-" * 50)


print("=== 模式2: Lingo 脚本 / 函数式风格 ===")

m = opt.Model()
x = m.add_var('X')
y = m.add_var('Y')
z = m.add_var('Z')

# Lingo Style
m.maximize(10 * x + 15 * y + 25 * z)
m.add(x + y + z <= 100)
m.add(x >= 10)
m.add(y >= 20)
m.add(z <= 30)

# 使用 PDHG 调度器求解
solver = opt.Solver(method='pdhg')
print("正在使用 PDHG 引擎求解...")
result_script = solver.solve(m)
print(f"最优化结果状态: {result_script.get('status')}")
print(f"最优解: {result_script.get('solution')}")
