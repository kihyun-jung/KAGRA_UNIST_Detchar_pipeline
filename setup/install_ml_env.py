#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import urllib.request
import tarfile
import shutil

# ================= Configuration =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Project Root

ENV_PREFIX = os.path.join(BASE_DIR, "ml_env")
MAMBA_ROOT = os.path.join(BASE_DIR, ".micromamba")
ENV_FILE = os.path.join(SCRIPT_DIR, "ml_environment.yml")

def get_system_config():
    """OS와 아키텍처를 감지하여 설정값을 반환 (Windows 미지원)"""
    system = platform.system().lower()   # 'darwin', 'linux'
    machine = platform.machine().lower() # 'arm64', 'x86_64' etc
    
    config = {
        "system": system,
        "is_apple_silicon": False,
        "target_platform": "",
        "micromamba_url": ""
    }

    # 1. macOS (Darwin)
    if system == "darwin":
        # Apple Silicon 감지
        if "arm" in machine or "aarch64" in machine:
            config["is_apple_silicon"] = True
            config["target_platform"] = "osx-arm64"
            mm_arch = "64" # Micromamba 바이너리는 Universal 호환
        else:
            # Intel Mac
            config["is_apple_silicon"] = False
            config["target_platform"] = "osx-64"
            mm_arch = "64"
        
        config["micromamba_url"] = f"https://micro.mamba.pm/api/micromamba/osx-{mm_arch}/latest"

    # 2. Linux
    elif system == "linux":
        config["is_apple_silicon"] = False
        config["target_platform"] = "linux-64"
        mm_arch = "aarch64" if "aarch64" in machine else "64"
        config["micromamba_url"] = f"https://micro.mamba.pm/api/micromamba/linux-{mm_arch}/latest"

    else:
        print(f"[!] Unsupported OS: {system}")
        print("    This pipeline supports macOS (Intel/Silicon) and Linux only.")
        sys.exit(1)
        
    return config

def setup_micromamba(url):
    """Micromamba 자동 다운로드 및 설치"""
    mamba_exe = os.path.join(MAMBA_ROOT, "micromamba")
    if os.path.exists(mamba_exe):
        return mamba_exe

    print(f"[*] Downloading Micromamba...")
    print(f"    URL: {url}")
    os.makedirs(MAMBA_ROOT, exist_ok=True)
    
    tar_path = os.path.join(MAMBA_ROOT, "mm.tar.bz2")
    
    try:
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path, "r:bz2") as tar:
            member = tar.getmember("bin/micromamba")
            member.name = "micromamba"
            tar.extract(member, path=MAMBA_ROOT)
        
        os.chmod(mamba_exe, 0o755)
        os.remove(tar_path)
        return mamba_exe
    except Exception as e:
        print(f"Failed to setup Micromamba: {e}")
        sys.exit(1)

def generate_env_yaml(config):
    print(f"[*] Generating YAML for {config['target_platform']} (Apple Silicon: {config['is_apple_silicon']})...")

    base_deps = """
name: ml_env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - scikit-learn
  - jupyterlab
  - tqdm
  - pytorch
  - torchvision
"""

    # [핵심] 아키텍처에 따른 TensorFlow 의존성 분기
    if config["is_apple_silicon"]:
        # Mac M1/M2/M3용 (Metal 가속 포함)
        tf_deps = """
  - pip:
    - hveto
    - gwpy
    - tensorflow-macos
    - tensorflow-metal
"""
    else:
        # Intel Mac & Linux용 (일반 TensorFlow)
        tf_deps = """
  - pip:
    - hveto
    - gwpy
    - tensorflow
"""

    full_yaml = base_deps.strip() + tf_deps.rstrip()

    with open(ENV_FILE, "w") as f:
        f.write(full_yaml)
    print(f"Created {ENV_FILE}")

def main():
    # 1. 시스템 감지
    config = get_system_config()
    
    # 2. Micromamba 준비
    mamba_exe = setup_micromamba(config["micromamba_url"])
    
    # 3. YAML 생성
    generate_env_yaml(config)
    
    # 4. 환경 생성
    print(f"\n [1/2] Creating Machine Learning environment in {ENV_PREFIX}...")
    print(f"      Target Platform: {config['target_platform']}")

    cmd = [
        mamba_exe, "create", 
        "-p", ENV_PREFIX, 
        "-r", MAMBA_ROOT,
        "-f", ENV_FILE,
        "-c", "conda-forge", "--yes",
        "--platform", config["target_platform"]
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n Error Occured: Installation (Exit code: {e.returncode})")
        sys.exit(1)

    # 5. 활성화 스크립트 생성
    activate_script = os.path.join(BASE_DIR, "activate_ml_env.sh")
    with open(activate_script, "w") as f:
        f.write("#!/bin/bash\n")
        # macOS의 경우 크로스 컴파일 방지를 위해 SUBDIR 명시
        if config["system"] == "darwin":
             f.write(f"export CONDA_SUBDIR={config['target_platform']}\n")
        
        f.write(f'SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n')
        f.write(f'eval "$("$SCRIPT_DIR/.micromamba/micromamba" shell hook --shell bash)"\n')
        f.write(f'micromamba activate "$SCRIPT_DIR/ml_env"\n')
        f.write('echo "Local Machine Learning Environment Activated!"\n')
        
        if config["is_apple_silicon"]:
            f.write('echo "MacOS Metal Acceleration Enabled"\n')
    
    os.chmod(activate_script, 0o755)
    
    if os.path.exists(ENV_FILE):
        os.remove(ENV_FILE)
    
    print(f"\n Setup Complete! Run: source ./activate_ml_env.sh")

if __name__ == "__main__":
    main()
