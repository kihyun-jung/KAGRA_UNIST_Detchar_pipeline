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
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Project Root

ENV_PREFIX = os.path.join(BASE_DIR, "python_env")
MAMBA_ROOT = os.path.join(BASE_DIR, ".micromamba")
IGWN_BASE_URL = "https://computing.docs.ligo.org/conda/environments"

def get_platform_info():
    system = platform.system().lower()
    machine = platform.machine().lower()
    return system, machine

def setup_micromamba(system, machine):
    mamba_exe = os.path.join(MAMBA_ROOT, "micromamba")
    if os.path.exists(mamba_exe):
        return mamba_exe

    print(f"[*] Downloading standalone Micromamba for {system}...")
    os.makedirs(MAMBA_ROOT, exist_ok=True)
    
    # [FIX] URL 생성 로직 수정 (darwin -> osx)
    if system == "darwin":
        platform_name = "osx"
        # IGWN 환경은 호환성을 위해 Mac M1/M2에서도 Intel(64) 버전을 강제함
        arch = "64" 
    else:
        platform_name = "linux"
        arch = "aarch64" if "arm" in machine else "64"

    url = f"https://micro.mamba.pm/api/micromamba/{platform_name}-{arch}/latest"
    tar_path = os.path.join(MAMBA_ROOT, "mm.tar.bz2")
    
    print(f"    Target URL: {url}") # 디버깅용 출력

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
        print(f"[!] Micromamba download failed: {e}")
        sys.exit(1)

def main():
    system, machine = get_platform_info()
    mamba_exe = setup_micromamba(system, machine)
    
    # Mac인 경우 항상 osx-64 사용 (IGWN 호환성)
    subdir = "osx-64" if system == "darwin" else "linux-64"
    target_url = f"{IGWN_BASE_URL}/{subdir}/igwn.txt"
    local_igwn_txt = os.path.join(SCRIPT_DIR, "igwn.txt")

    print(f"[*] Downloading requirement file: {target_url}")
    try:
        req = urllib.request.Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(local_igwn_txt, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Successfully downloaded igwn.txt to local.")
    except Exception as e:
        print(f"Failed to download igwn.txt: {e}")
        sys.exit(1)

    print(f"\n [1/2] Creating isolated environment via Micromamba...")
    cmd = [
        mamba_exe, "create", 
        "-p", ENV_PREFIX, 
        "-r", MAMBA_ROOT,
        "-f", local_igwn_txt, 
        "-c", "igwn", "-c", "conda-forge", "--yes",
        "--platform", subdir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n Error Occured: Installation(Exit code: {e.returncode})")
        sys.exit(1)

    activate_script = os.path.join(BASE_DIR, "activate_igwn_env.sh")
    with open(activate_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export CONDA_SUBDIR={subdir}\n")
        f.write(f'SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n')
        f.write(f'eval "$("$SCRIPT_DIR/.micromamba/micromamba" shell hook --shell bash)"\n')
        f.write(f'micromamba activate "$SCRIPT_DIR/python_env"\n')
        f.write('echo "Local IGWN Environment Activated!"\n')
    
    os.chmod(activate_script, 0o755)
    
    if os.path.exists(local_igwn_txt):
        os.remove(local_igwn_txt)
    
    print(f"\n Setup Complete! Command: source ./activate_igwn_env.sh")

if __name__ == "__main__":
    main()
