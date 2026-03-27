import threading
import time
import sys
import os
import subprocess

import webview

# 导入您现有的后端代码
from server import PORT, LingoHTTPRequestHandler
import http.server
import socketserver

def start_background_server():
    # 确保运行路径正确，使得 static 文件夹能被找到
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    socketserver.TCPServer.allow_reuse_address = True
    
    # 构建一个静默的 Handler 避免控制台一直刷屏
    class QuietHandler(LingoHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
            
    try:
        httpd = socketserver.ThreadingTCPServer(("", PORT), QuietHandler)
        httpd.serve_forever()
    except Exception as e:
        print(f"Server backend error: {e}")

if __name__ == '__main__':
    print("⚡ 正在启动 UltimateSolver 核心引擎...")
    
    # 1. 在后台独立线程中启动我们的 Python 求解计算服务
    server_thread = threading.Thread(target=start_background_server, daemon=True)
    server_thread.start()
    
    # 2. 稍微等待确保本地端口 8000 开放
    time.sleep(1)
    
    # 3. 呼出系统级独立应用程序窗口 (Windows 环境下默认调用 Edge WebView2，具备顶级现代浏览器性能)
    window = webview.create_window(
        title='UltimateSolver 极速异构求解工作站',
        url=f'http://127.0.0.1:{PORT}',
        width=1280,
        height=850,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
        confirm_close=True,
        background_color='#0f111a'  # 完美契合您的无缝暗黑UI
    )
    
    print("🚀 正在渲染本地原生客户端视窗...")
    # 开启硬件加速渲染的独立窗口
    webview.start(private_mode=False)
