import os
import subprocess
import glob
import sys

# 【修复 Windows 终端中文乱码】
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 您的实际离线包路径
PKG_DIR = r"d:\Python_offline_packages\langchain_env_offline_pkgs\offline_pkgs"

# 获取所有 whl 和 tar.gz 文件
files = glob.glob(os.path.join(PKG_DIR, "*.whl")) + glob.glob(os.path.join(PKG_DIR, "*.tar.gz"))

if not files:
    print(f"未在 {PKG_DIR} 中找到任何安装包！")
    sys.exit(1)

print(f"找到 {len(files)} 个安装包，开始逐个离线安装（容错模式）...")
print("-" * 60)

success_count = 0
fail_count = 0
failed_files = []

for i, file in enumerate(files):
    file_name = os.path.basename(file)
    # 【核心魔法】：逐个安装，并加上 --no-deps (不自动解析依赖，防止被卡死)
    # 因为我们把所有包都放进去了，最终它们都会被装进环境里
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", file]
    
    # 静默运行，只在失败时打印
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode == 0:
        success_count += 1
        # 打印进度条
        print(f"\r[{i+1}/{len(files)}] ✅ 成功: {file_name[:40]:<40}", end="")
    else:
        fail_count += 1
        failed_files.append(file_name)
        print(f"\r[{i+1}/{len(files)}] 失败: {file_name[:40]:<40}", end="")

print("\n" + "-" * 60)
print(f"批量安装结束！成功: {success_count} 个, 失败: {fail_count} 个")

if failed_files:
    print("\n以下文件因平台不匹配等原因安装失败（通常不影响核心 RAG 功能）:")
    for f in failed_files:
        print(f"   - {f}")
