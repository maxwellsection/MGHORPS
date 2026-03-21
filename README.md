<div align="center">

# ⚡ UltimateSolver
**A High-Performance, Heterogeneous Hardware Accelerated LP/MILP Solver in Pure Python**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Hardware Accelerators](https://img.shields.io/badge/Accelerators-Vulkan%20%7C%20CUDA%20%7C%20NPU-orange)]()

[English](README_EN.md) | 简体中文

</div>

`UltimateSolver` 是一个新的、突破性的运筹学（OR）求解器框架。它在不依赖任何庞大的 C++ 商业求解器后端（如 Gurobi, CPLEX）的情况下，依靠革命性的**多端异构硬件加速（GPU / Vulkan / NPU）**与现代化的**单纯形法、分支定界算法迭代**，在纯 Python 环境下实现了工业级的数学规划求解性能。

此外，它还配备了如同 **PyTorch 般丝滑** 的全新面向对象 Python API 接口，以及高度兼容老旧系统生态的内嵌 **Lingo 解析与编译器**。

---

## 🌟 核心理念与技术突破 (Key Features)

- 🚀 **PyTorch-Style 现代建模 API (`ultimate_opt`)**
  告别晦涩难懂的传统矩阵输入。采用类似深度学习框架自动微分图的理念设计，变量、约束和目标函数都是可直观操作和组合的 Python 对象，代码可读性极高。
- ⚡ **原生异构加速引擎 (Heterogeneous Accelerators)**
  囊括了当下最前沿的计算方案：
  - 基于原对偶混合梯度法 (**PDHG**) 的超大规模并行求解器。
  - **Vulkan Compute Shader** 支持，打破 CUDA 独占，实现全平台 GPU 计算。
  - **NPU Edge Scheduler**：面向边缘端 NPU 设备特化的算子极切分技术 (Nano-slicing)。
- 🧮 **纯纯的高级算法内核 (Advanced Algorithmic Core)**
  - **自适应稀疏修正单纯形 (Adaptive Sparse Revised Simplex)**，处理高度稀疏大模型更快捷。
  - **强健两阶段单纯形法** 保障了复杂初始可行基的寻找和数值稳定性。
  - 原生 **MILP (混合整数线性规划)** 的分支定界求解核心。
  - 自带数学模型空间**降维预处理器 (Presolver)**。
- 🛠️ **一体化编译与 IDE (Built-in IDE & Compiler)**
  零配置自带基于控制台/UI 的 Lingo 语法解析编译器与运行环境，方便工业级传统模型无缝迁移。

---

## 📦 安装说明

本项目去除了冗杂的历史包袱，您可以直接克隆或将整个 `lp_solver_core_export` 文件夹作为依赖库集成进您的项目中：

```bash
# 1. 克隆底层引擎核心
git clone https://github.com/your-username/ultimate-solver-core.git

# 2. 安装建议的基础依赖 (如 numpy, 还有可选的 vulkan 或 cupy 模块)
pip install numpy
```

---

## ⏱️ 快速上手: PyTorch 风格建模 (Quick Start)

通过直观的面向对象 API 实现您的第一个工厂生产优化模型：

```python
from ultimate_opt import Model, Variable, Constraint

# 1. 初始化模型上下文
model = Model("Factory Planning Optimization")

# 2. 定义变量 (支持 continuous, binary, integer 等)
x1 = Variable(name="Product_A", var_type="continuous", lb=0)
x2 = Variable(name="Product_B", var_type="continuous", lb=0)

# 3. 添加带有重载操作符的约束条件
model.add_constraint( 2 * x1 + x2 <= 100, name="Machine_Time_Limit" )
model.add_constraint( x1 + 2 * x2 <= 80, name="Labor_Time_Limit" )

# 4. 设置优化目标 (最大化利润)
model.set_objective( 40 * x1 + 30 * x2, sense="maximize" )

# 5. 一键编译并自动选择底层硬件引擎进行求解
solution = model.solve()

print("Status:", solution.status)
print("Optimal X1:", solution.get_value(x1))
print("Optimal X2:", solution.get_value(x2))
print("Max Profit:", solution.objective_value)
```

---

## 🗂️ 项目架构概览

```text
ultimate-solver-core/
├── ultimate_opt.py                 # PyTorch 风格的高级封装入口 (Recommended)
├── ultimate_solvers_unified.py     # 底层架构的统一硬件调度分发网关
├── presolver.py                    # 建模级别的数学冗余消元与变量降维器
├── 核心求解内核 / 
│   ├── ultimate_lp_solver.py       # 核心双相单纯形 LP 求解基石
│   ├── milp_solver.py              # 分支定界 MILP 求解器
│   └── sparse_revised_simplex.py   # 自适应稀疏化单纯形修正
├── 异构并行前端 / 
│   ├── pdhg_accelerated_solver.py  # Primal-Dual Hybrid Gradient 高并行求解器
│   ├── vulkan_compute_accelerator.py # 基于 Vulkan 的跨平台底层操作
│   ├── multi_gpu_accelerated_solver.py # 面向数据中心的多显卡协同 
│   └── npu_edge_scheduler.py       # 面向边缘计算的 NPU 切分调度策略
└── 传统语法与读写兼容支持 /
    ├── lingo_compiler.py / lingo_ide.py 
    └── lp_reader.py / mps_reader.py
```

---

## 📚 说明文档与进阶教程

更完整的全方位介绍、多硬件调度策略切换、模型调优与 Lingo 文本解析的高级操作手册，请参阅随附的 **使用说明书**:

👉 [**🔗 点击查看 《UltimateSolver 用户操作与进阶使用手册》 (USER_MANUAL.md)**](USER_MANUAL.md)

---

## 🤝 参与贡献 (Contributing)

任何关于算法调优、GPU 算子优化甚至新颖硬件的加速支持，我们都非常欢迎！请阅读 `CONTRIBUTING.md` 并提交您的 Pull Request！

## 📄 许可证 (License)
本项目遵守 [MIT](LICENSE) 开源协议。
