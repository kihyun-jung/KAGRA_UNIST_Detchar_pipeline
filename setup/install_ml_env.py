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

def is_apple_silicon_check():
    """
    단순 platform.machine() 확인이 아니라, 
    Rosetta 뒤에 숨은 실제 하드웨어가 Apple Silicon인지 확인합니다.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "darwin":
        return False

    # 1. 네이티브 ARM 모드인 경우
    if "arm" in machine or "aarch64" in machine:
        return True

    # 2. x86_64로 뜨지만 실제로는 Rosetta로 돌고 있는 M1/M2/M3인 경우
    try:
        # sysctl 명령어로 번역(Rosetta) 여부 확인
        # sysctl.proc_translated가 1이면 Rosetta 환경임 -> 즉, 실제는 Apple Silicon
        result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], 
                                capture_output=True, text=True)
        if result.stdout.strip() == "1":
            print("[!] Detected Apple Silicon running under Rosetta (x86_64 emulation).")
            print("    -> Forcing Native ARM64 installation for performance.")
            return True
    except Exception:
        pass

    return False

def get_system_config():
    """OS와 아키텍처를 감지하여 설정값을 반환"""
    system = platform.system().lower()
    
    config = {
        "system": system,
        "is_apple_silicon": False,
        "target_platform": "",
        "micromamba_url": ""
    }

    # 1. macOS (Darwin)
    if system == "darwin":
        if is_apple_silicon_check():
            config["is_apple_silicon"] = True
            config["target_platform"] = "osx-arm64"
            mm_arch = "64" # Micromamba 바이너리 다운로드용 (범용)
        else:
            # 찐 Intel Mac
            config["is_apple_silicon"] = False
            config["target_platform"] = "osx-64"
            mm_arch = "64"
        
        config["micromamba_url"] = f"https://micro.mamba.pm/api/micromamba/osx-{mm_arch}/latest"

    # 2. Linux
    elif system == "linux":
        machine = platform.machine().lower()
        config["is_apple_silicon"] = False
        config["target_platform"] = "linux-64"
        mm_arch = "aarch64" if "aarch64" in machine else "64"
        config["micromamba_url"] = f"https://micro.mamba.pm/api/micromamba/linux-{mm_arch}/latest"

    else:
        print(f"[!] Unsupported OS: {system}")
        print("    This pipeline supports macOS (Intel/Silicon) and Linux.")
        sys.exit(1)
        
    return config

def setup_micromamba(url):
    mamba_exe = os.path.join(MAMBA_ROOT, "micromamba")
    if os.path.exists(mamba_exe):
        return mamba_exe

    print(f"[*] Downloading Micromamba...")
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
    print(f"[*] Generating YAML for {config['target_platform']}...")

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

    if config["is_apple_silicon"]:
        # Mac M1/M2/M3 (nomkl 추가하여 인텔 라이브러리 배제)
        tf_deps = """
  - nomkl
  - pip:
    - hveto
    - gwpy
    - tensorflow-macos
    - tensorflow-metal
"""
    else:
        # Intel Mac & Linux
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
    # 1. 스마트 시스템 감지 (Rosetta 관통)
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
        "--platform", config["target_platform"] # 플랫폼 강제 지정이 핵심
    ]

    try:
        # 환경 변수에도 플랫폼 강제 주입 (Rosetta 무력화)
        env = os.environ.copy()
        if config["target_platform"]:
            env["CONDA_SUBDIR"] = config["target_platform"]
            
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n Error Occured: Installation (Exit code: {e.returncode})")
        sys.exit(1)

    # 5. 활성화 스크립트 생성
    activate_script = os.path.join(BASE_DIR, "activate_ml_env.sh")
    with open(activate_script, "w") as f:
        f.write("#!/bin/bash\n")
        # 활성화 시에도 아키텍처 고정
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
