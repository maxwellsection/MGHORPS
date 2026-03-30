# 📖 UltimateSolver 用户操作与进阶使用手册

欢迎使用 **UltimateSolver 核心引擎**！
本手册旨在为您提供该运筹优化求解框架的详细使用指导，包括全新的 PyTorch 式快速建模 (`ultimate_opt.py`)，底层多硬件后端调度机制，及基于 Lingo 脚本的兼容用法。

---

## 一、系统架构与模块 (Architecture overview)

UltimateSolver 并非单一算法的黑盒，而是一个灵活的可替换式架构栈。所有对外的调用主要通过以下几个网关暴露给用户：
1. **`ultimate_opt.py`**：当前**最推荐**的用户入口点。通过 `Model`, `Variable`, `Constraint` 构建数学表达树，系统会通过抽象语法分析，自动将其转化为后端接受的标准形式（如系数矩阵）并下达任务。
2. **`ultimate_solvers_unified.py`**：提供 `solve_problem()` 函数，用于接收标准化数据格式（字典，包含 objective, constraints）并调度到对应的求解内核。
3. **`lingo_ide.py`**：交互式的文本建模编译器前端。

---

## 二、基础教程：PyTorch 式优雅建模 (Building a Model)

借助 `ultimate_opt.py`，再也不用手动拼写繁琐的 `A, b, c, lb, ub` 等矩阵。

### 1. 建立模型上下文
```python
from ultimate_opt import Model, Variable

# 参数包含模型名称以及求解容差等可选超参
prob = Model("Investment_Optimization", tolerance=1e-8)
```

### 2. 添加决策变量 (Variables)
你可以像定义常规张量那样定义决策变量。
```python
# 连续非负变量
x = Variable(name="Bond_X", var_type="continuous", lb=0, ub=None)

# 0-1 二进制变量
y = Variable(name="Select_Project_Y", var_type="binary")

# 整数自由变量
z = Variable(name="Workers_Z", var_type="integer", lb=-10, ub=10)
```

### 3. 添加约束条件 (Constraints)
系统支持 `+ - * /` 的 Python 原生操作符重载机制，自动解析数学表达式。
```python
# 直接利用 Python 符号添加约束
prob.add_constraint( 3 * x + 5 * z <= 100, name="Budget_Constraint" )
prob.add_constraint( y + x >= 2, name="Minimum_Selection" )
prob.add_constraint( 2 * y == 1, name="Exact_Condition" )
```

### 4. 设定目标函数并一键求解 (Objective & Solve)
```python
prob.set_objective( 10 * x + 20 * y + 5 * z, sense="maximize" )

# 调用 solve() 时，引擎会:
# 1. 调用 presolver.py 压缩模型
# 2. 判断是否存在非连续变量，来调度 milp_solver 或 ultimate_lp_solver
# 3. 询问 ultimate_solvers_unified，并根据硬件配置下放
solution = prob.solve()
```

### 5. 获取并打印结果 (Extracting Results)
```python
if solution.status == 'optimal':
    print(f"最大收益: {solution.objective_value}")
    print(f"投向 Bond_X 的资金: {solution.get_value(x)}")
    # 获取整个变量的字典表
    print(f"所有变量分布: {solution.variables}")
else:
    print(f"无解/无界 解状态: {solution.status}, Details: {solution.message}")
```

---

## 三、进阶：多硬件异步加速与切换 (Hardware acceleration setup)

得益于新增加的 `vulkan_compute_accelerator.py`, `pdhg_accelerated_solver.py` 等模块，您可以非常方便地切换到底层加速库获取成倍性能。

目前，推荐的方式是在初始化 `Model` 或者调用统一后端时指定参数。不过在底层 API 中，通常是如下方法工作：

### 1. Vulkan 跨平台 GPU 加速
适用于任何装有带有 Vulkan 驱动的独立/集成显卡的设备（Windows/Linux）：
```python
# 引擎底层触发示例（一般无需用户手动干预）：
from vulkan_compute_accelerator import VulkanComputeAccelerator
accel = VulkanComputeAccelerator()
result_array = accel.matrix_multiply(A, x) 
```
如果您的目标模型规模过大，系统会自动切换到 `multi_gpu_accelerated_solver` 或借助其 Vulkan 实现降维。

### 2. NPU 边缘切片加速
专为大边缘网关设备设计：
如果您所在的运行设备包含类似昇腾 / 高通平台等 NPU，系统内的 `npu_edge_scheduler.py` 支持 "Nano-slicing"，即对系数矩阵进行小片切割并发放指令：
```python
# 开启 NPU 特化选项：
from npu_edge_scheduler import NpuEdgeScheduler
scheduler = NpuEdgeScheduler(num_npu=2)
# 分配并监控
```

---

## 四、高级算法特性：预处理与稀疏优化

大公司的模型可能包含上百万条约束，其中大部分是无效边界。引擎提供了非常强悍的模型前处理逻辑以适应工业落地。

### 1. 模型预处理引擎 (`presolver.py`)
主要功能：
- **空行/列移除**
- **常量替换**
- **重复约束探测/降阶**
在调用 `solve()` 时可强制开启或关闭：
```python
# 默认 auto 检测开启，可以手动禁用以观测原始求解树
solution = model.solve(enable_presolve=True)
```

### 2. 稀疏修正单纯形法 (`sparse_revised_simplex.py`)
传统的表格单纯形法 (Tableau Simplex) 在 $O(N^2)$ 的复杂度下进行旋转。而稀疏修正方法只持有基础逆矩阵 $\mathbf{B}^{-1}$ 的稀疏更新信息。这极大地拓展了系统在面对拥有大量对角线 `0` 的极稀疏网络流等模型的处理能力。

---

## 五、传统工业标准：解析器与 Lingo 兼容模块

如果您已经积累了大量过去工业上产生的历史模型文件（`.mps` 或 `.lp` 或 `.lng` 文档），没关系，系统完美支持。

### 1. 数据标准格式流读取 (`lp_reader`, `mps_reader`)
```python
from mps_reader import parse_mps

model_dict = parse_mps("large_transportation_problem.mps")

# model_dict 是一套系统字典，可直接下发至统一网关：
from ultimate_solvers_unified import solve_problem
result = solve_problem(model_dict)
print(result['objective_value'])
```

### 2. Lingo 脚本零门槛编译 (`lingo_compiler`)
利用我们的 Lingo Compiler，只要输入标准的 Lingo 脚本，即可解析求解：
```python
from lingo_compiler import LingoCompiler

script = """
MODEL:
MAX = 20 * X + 30 * Y;
X + Y <= 100;
2*X + 4*Y <= 240;
END
"""
compiler = LingoCompiler()
result_dict = compiler.compile_and_solve(script)
print(result_dict)
```

如果您习惯 GUI 界面交互，还可以运行随文件夹导出的 `lingo_ide.py` 开启控制台前端窗口，所见即所得。

---

## 💡 常见故障排除 (Troubleshooting)

1. **`ValueError: non-scalar numpy.ndarray cannot be used for fill`**  
   这是我们在早期两阶段单纯形开发中经常遇见的类型系统异常。最新版本已经完全修复，如果您使用了外置的自定义约束注入（即 `coeffs` 是非平铺的 numpy 数组），请确保通过 `flatten()` 或使用全新 `ultimate_opt.py` 原生封装规避。
   
2. **"Unbounded" (无界解) 经常误报？**  
   我们在此导出版本中优化了 **Variable Bounds 追踪解析器** 的机制。请务必优先通过 `Variable(lb=..., ub=...)` 声明而不是使用 `<0`, `>0` 隐式约束声明变量极值，可以大幅提升预处理器 (`presolver`) 边界剪枝速度。
   
3. **Conda 或环境抛出缺少 `cupy`**  
   GPU 加速模块（`multi_gpu...`）虽然非常快，但如果未检测到合理的显卡驱动和 Python 桥接（如 CuPy），会自动 Fallback （回退）为 CPU 多线程。不会阻断模型运行。
   
如有其它问题或发现新 Bug，欢迎提报并在 Issue 中沟通交流！
