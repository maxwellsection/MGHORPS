import os
import subprocess
import glob
import sys

# 设置标准输出为 UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

env = os.environ.copy()
# Insert cmake path
env['PATH'] = r"D:\c make\bin;" + env['PATH']

# MinGW compiler usually resides in Qt's Tools folder
mingw_paths = glob.glob(r'D:\qt\Tools\mingw*\bin')
if mingw_paths:
    env['PATH'] = mingw_paths[-1] + ";" + env['PATH']
    print(f"[OK] Found MinGW at: {mingw_paths[-1]}")
else:
    print(r"Not found MinGW in D:\qt\Tools, relying on global scope.")

print("====================================")
print("[1/2] Running CMake configure...")
cmd_config = [r'D:\c make\bin\cmake.exe', '-B', 'build', '-G', 'MinGW Makefiles', '-DCMAKE_PREFIX_PATH=D:\\qt\\6.11.0\\mingw_64']

res_config = subprocess.run(cmd_config, env=env, cwd=r'd:\githubtset\qt_gui', capture_output=True, text=True)
print(res_config.stdout)

if res_config.returncode != 0:
    print("[ERROR] CMake Configure failed:")
    print(res_config.stderr)
else:
    print("[OK] CMake Configure success!")
    print("====================================")
    print("[2/2] Running CMake build...")
    
    cmd_build = [r'D:\c make\bin\cmake.exe', '--build', 'build']
    res_build = subprocess.run(cmd_build, env=env, cwd=r'd:\githubtset\qt_gui', capture_output=True, text=True)
    print(res_build.stdout)
    
    if res_build.returncode != 0:
        print("[ERROR] Build failed:")
        print(res_build.stderr)
    else:
        print("[SUCCESS] Build 100% complete! Executable is in the build directory.")

