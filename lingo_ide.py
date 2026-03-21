import sys
import json
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTextEdit, QPushButton, QSplitter, 
                               QLabel, QToolBar, QComboBox, QDialog, QFormLayout, 
                               QDialogButtonBox)
from PySide6.QtCore import Qt, QThread, Signal
from lingo_compiler import LingoCompiler

# Try importing solver
try:
    from ultimate_lp_solver import UltimateLPSolver
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False

class SolverThread(QThread):
    output_signal = Signal(str)
    finished_signal = Signal(dict)
    
    def __init__(self, code_text, config):
        super().__init__()
        self.code_text = code_text
        self.config = config
        
    def run(self):
        try:
            # 1. Compile
            self.output_signal.emit(">>> 开始解析 LINGO 代数模型...")
            compiler = LingoCompiler()
            model = compiler.compile(self.code_text)
            
            n_vars = len(model['variables'])
            n_cons = len(model['constraints'])
            self.output_signal.emit(f">>> 编译成功: {n_vars} 个变量, {n_cons} 个约束条件")
            
            if not HAS_SOLVER:
                self.output_signal.emit(">>> [错误] 找不到 ultimate_lp_solver.py 核心引擎，请检查安装环境！")
                self.finished_signal.emit({'status': 'error'})
                return
                
            # 2. Config & Initialization
            backend = self.config.get('backend', 'cpu')
            self.output_signal.emit(f">>> 初始化终极求解器引擎...计算后端: {backend.upper()}")
            
            # Map backend to args
            solver_type = 'auto'
            use_npu = False
            if backend == 'vulkan':
                solver_type = 'revised_simplex' # Or dedicated vulkan
            elif backend == 'cuda':
                solver_type = 'pdhg'
            elif backend == 'npu':
                use_npu = True
                
            solver = UltimateLPSolver(solver=solver_type, use_npu=use_npu)
            
            # 3. Solve 
            self.output_signal.emit(">>> 开始数学优化...")
            res = solver.solve(model['objective'], model['constraints'], model['variables'])
            
            self.output_signal.emit(">>> ======================================")
            self.output_signal.emit(f">>> 优化计算结束! 耗时: {res.get('solve_time', 0):.4f}秒")
            self.output_signal.emit(f">>> 最终状态: {res.get('status', 'Unknown')}")
            
            if res.get('status') == 'optimal':
                self.output_signal.emit(f">>> 最优目标值: {res.get('objective_value')}")
                # Print non-zero variables natively
                sol = res.get('solution', [])
                for i, v in enumerate(sol):
                    if abs(v) > 1e-6:
                        v_name = model['variables'][i]['name']
                        self.output_signal.emit(f"    {v_name} = {v:.6f}")
            else:
                self.output_signal.emit(f">>> 模型无解或无界: {res.get('message', '')}")
                
            self.finished_signal.emit(res)
            
        except Exception as e:
            self.output_signal.emit(f"\\n>>> [致命错误]: 求解过程异常: {str(e)}")
            self.finished_signal.emit({'status': 'error'})

class SettingsDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("硬件加速设置")
        self.resize(300, 150)
        
        self.config = current_config
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["cpu", "cuda", "vulkan", "opengl", "npu"])
        
        # Set current backend
        idx = self.backend_combo.findText(self.config.get('backend', 'cpu'))
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
            
        form.addRow("目标计算硬件后端:", self.backend_combo)
        layout.addLayout(form)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_config(self):
        return {
            'backend': self.backend_combo.currentText()
        }

class LingoIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenOR Desktop IDE (仿LINGO通用版)")
        self.resize(1000, 700)
        
        self.config = self.load_config()
        self.solver_thread = None
        self.init_ui()
        
    def load_config(self):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except Exception:
            return {'backend': 'cpu'}
            
    def save_config(self):
        with open("config.json", "w") as f:
            json.dump(self.config, f)
            
    def init_ui(self):
        # Toolbar
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        
        run_btn = QPushButton("🚀 运行求解 (Solve)")
        run_btn.setStyleSheet("font-weight: bold; color: green;")
        run_btn.clicked.connect(self.run_solver)
        toolbar.addWidget(run_btn)
        
        settings_btn = QPushButton("⚙️ 硬件设置")
        settings_btn.clicked.connect(self.open_settings)
        toolbar.addWidget(settings_btn)
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Editor Pane
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.addWidget(QLabel("📝 模型代码编辑器 (类 LINGO 语法):"))
        self.editor = QTextEdit()
        self.editor.setStyleSheet("font-family: Consolas, monospace; font-size: 14px;")
        
        # Add default template
        self.editor.setPlainText("! 在此处输入运筹代数模型 ;\nMAX = 20*X1 + 30*X2;\nSUBJECT TO;\n  X1 + X2 <= 50;\n  3*X1 + 2*X2 <= 100;\n@FREE(X1);\n")
        
        editor_layout.addWidget(self.editor)
        splitter.addWidget(editor_widget)
        
        # Console Pane
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.addWidget(QLabel("🖥️ 求解器输出终端:"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #1e1e1e; color: #00ff00;")
        console_layout.addWidget(self.console)
        splitter.addWidget(console_widget)
        
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter)
        
    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.config = dlg.get_config()
            self.save_config()
            self.append_console(f"已应用设置并保存 config.json: 后端切换为 {self.config['backend'].upper()}")
            
    def append_console(self, text):
        self.console.append(text)
        # Scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def run_solver(self):
        if self.solver_thread and self.solver_thread.isRunning():
            self.append_console(">>> 求解引擎已经在运行中！请稍候。")
            return
            
        code = self.editor.toPlainText()
        self.console.clear()
        
        self.solver_thread = SolverThread(code, self.config)
        self.solver_thread.output_signal.connect(self.append_console)
        self.solver_thread.finished_signal.connect(self.on_solve_finished)
        self.solver_thread.start()
        
    def on_solve_finished(self, result_dict):
        self.append_console("\n>>> 处理完毕。就绪。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Optional styling
    app.setStyle("Fusion")
    
    window = LingoIDE()
    window.show()
    sys.exit(app.exec())
