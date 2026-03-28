#!/usr/bin/env python3
"""
LP Solver 演示界面 (Streamlit 版本)
用于展示 githubtset 文件夹中线性规划求解器的功能
"""

import streamlit as st
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入求解器
try:
    from ultimate_lp_solver import UltimateLPSolver
    SOLVER_AVAILABLE = True
except ImportError as e:
    SOLVER_AVAILABLE = False
    st.error(f"无法导入求解器: {e}")

st.set_page_config(page_title="OpenOR LP Solver 演示", layout="wide")

st.title("🚀 OpenOR 线性规划求解器")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 求解器设置")
    
    solver_type = st.selectbox(
        "求解器类型",
        ["auto", "revised_simplex", "pdhg", "pulp"],
        help="选择求解算法"
    )
    
    use_gpu = st.checkbox("使用 GPU 加速", value=False)
    
    tolerance = st.slider("数值容差", 1e-10, 1e-6, 1e-8, format="%.0e")
    
    st.markdown("---")
    st.info("💡 这是一个用于演示的简化界面")

# 主界面
tab1, tab2, tab3 = st.tabs(["📋 问题建模", "🔢 示例问题", "📊 结果分析"])

with tab1:
    st.subheader("定义线性规划问题")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**目标函数**")
        obj_type = st.radio("优化类型", ["minimize", "maximize"], horizontal=True)
        obj_coeffs = st.text_area(
            "目标函数系数 (用逗号分隔)",
            value="3, 2, -1, 5",
            help="例如: 3x1 + 2x2 - x3 + 5x4"
        )
    
    with col2:
        st.write("**约束条件**")
        constraints_input = st.text_area(
            "约束条件 (每行一个)",
            value="""<= 2 1 0 1 10
<= 1 0 2 1 8
=  1 1 1 0 5""",
            help="格式: [类型] [系数...] [右端项]"
        )
    
    if st.button("🚀 求解", type="primary"):
        if not SOLVER_AVAILABLE:
            st.error("求解器未正确加载")
        else:
            with st.spinner("正在求解..."):
                try:
                    # 解析目标函数
                    c = [float(x.strip()) for x in obj_coeffs.split(",")]
                    n_vars = len(c)
                    
                    objective = {
                        'type': obj_type,
                        'coeffs': c
                    }
                    
                    # 解析约束
                    constraints = []
                    for line in constraints_input.strip().split("\n"):
                        parts = line.split()
                        if len(parts) >= 3:
                            ctype = parts[0]
                            coeffs = [float(x) for x in parts[1:-1]]
                            rhs = float(parts[-1])
                            constraints.append({
                                'type': ctype,
                                'coeffs': coeffs,
                                'rhs': rhs
                            })
                    
                    # 定义变量
                    variables = [{'name': f'x{i+1}', 'type': 'nonneg'} for i in range(n_vars)]
                    
                    # 创建求解器并求解
                    solver = UltimateLPSolver(
                        tolerance=tolerance,
                        solver=solver_type,
                        use_gpu=use_gpu
                    )
                    
                    result = solver.solve(objective, constraints, variables)
                    
                    # 显示结果
                    st.success(f"求解完成！状态: {result.get('status', 'unknown')}")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("最优值", f"{result.get('objective_value', 0):.4f}")
                    res_col2.metric("求解时间", f"{result.get('solve_time', 0)*1000:.2f} ms")
                    res_col3.metric("迭代次数", result.get('iterations', 0))
                    
                    # 显示解
                    st.write("**最优解:**")
                    solution = result.get('solution', [])
                    sol_data = {f"x{i+1}": float(solution[i]) for i in range(len(solution))}
                    st.bar_chart(sol_data)
                    
                    # 详细数值
                    with st.expander("查看详细数值"):
                        for i, v in enumerate(solution):
                            st.write(f"x{i+1} = {v:.6f}")
                    
                except Exception as e:
                    st.error(f"求解失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab2:
    st.subheader("预设示例问题")
    
    example = st.selectbox(
        "选择示例",
        ["生产计划问题", "运输问题", " diet 问题 (营养搭配)"]
    )
    
    if example == "生产计划问题":
        st.markdown("""
        **问题描述**: 某工厂生产两种产品 A 和 B
        - 产品 A 利润: 3元/件
        - 产品 B 利润: 2元/件
        - 机器工时约束: 2A + B ≤ 10
        - 人工约束: A + 2B ≤ 8
        - 非负约束: A, B ≥ 0
        """)
        
        if st.button("运行此示例"):
            if SOLVER_AVAILABLE:
                solver = UltimateLPSolver(solver=solver_type)
                
                objective = {'type': 'maximize', 'coeffs': [3, 2]}
                constraints = [
                    {'type': '<=', 'coeffs': [2, 1], 'rhs': 10},
                    {'type': '<=', 'coeffs': [1, 2], 'rhs': 8}
                ]
                variables = [{'name': 'A', 'type': 'nonneg'}, {'name': 'B', 'type': 'nonneg'}]
                
                result = solver.solve(objective, constraints, variables)
                
                st.success(f"最优利润: {result.get('objective_value', 0):.2f} 元")
                sol = result.get('solution', [0, 0])
                st.write(f"生产 A: {sol[0]:.2f} 件")
                st.write(f"生产 B: {sol[1]:.2f} 件")

with tab3:
    st.subheader("求解结果分析")
    st.info("求解完成后，此处将显示详细的敏感性分析和可视化结果")
    
    # 可以添加更多分析图表
    st.write("功能开发中...")

st.markdown("---")
st.caption("OpenOR LP Solver Demo | 基于 githubtset/ultimate_lp_solver.py")
