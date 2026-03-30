import http.server
import socketserver
import json
import time

# 尝试导入核心求解环境
from lingo_compiler import LingoCompiler
try:
    from ultimate_lp_solver import UltimateLPSolver
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False

PORT = 8000

class LingoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # 强制 serve 当前目录下的 static 文件夹内文件
        super().__init__(*args, directory="static", **kwargs)

    def do_POST(self):
        if self.path == '/api/solve':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                code_text = data.get('code', '')
                backend = data.get('backend', 'cpu')
                
                # 编译前端传过来的文字
                start_compile = time.time()
                compiler = LingoCompiler()
                model = compiler.compile(code_text)
                compile_time = time.time() - start_compile
                
                if not HAS_SOLVER:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': '找不到核心引擎 ultimate_lp_solver.py, 请检查完整性！'}).encode('utf-8'))
                    return
                
                # 配置计算硬件类型
                solver_type = 'auto'
                use_npu = False
                if backend == 'vulkan':
                    solver_type = 'revised_simplex'
                elif backend == 'cuda':
                    solver_type = 'pdhg'
                elif backend == 'npu':
                    use_npu = True
                    
                # 运行求解
                solver = UltimateLPSolver(solver=solver_type, use_npu=use_npu)
                start_solve = time.time()
                res = solver.solve(model['objective'], model['constraints'], model['variables'])
                solve_time = time.time() - start_solve
                
                # 组装返回结果
                response = {
                    'status': res.get('status', 'Unknown'),
                    'objective_value': res.get('objective_value', None),
                    'solution': [],
                    'compile_time': compile_time,
                    'solve_time': res.get('solve_time', solve_time),
                    'num_vars': len(model['variables']),
                    'num_cons': len(model['constraints']),
                    'message': res.get('message', '')
                }
                
                if response['status'] == 'optimal':
                    sol = res.get('solution', [])
                    for i, v in enumerate(sol):
                        if abs(v) > 1e-6:
                            v_name = model['variables'][i]['name']
                            response['solution'].append({'name': v_name, 'value': float(v)})
                            
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    # 以防端口复用抛错，设置 allow_reuse_address
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), LingoHTTPRequestHandler) as httpd:
        print(f"==========================================")
        print(f" UltimateSolver UI Server Started!        ")
        print(f" Serving at: http://127.0.0.1:{PORT}      ")
        print(f"==========================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\\nServer stopped.")
