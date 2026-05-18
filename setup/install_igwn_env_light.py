#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import urllib.request
import tarfile

# ================= Configuration =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Project Root

ENV_PREFIX = os.path.join(BASE_DIR, "python_env")
MAMBA_ROOT = os.path.join(BASE_DIR, ".micromamba")

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
    
    if system == "darwin":
        platform_name = "osx"
        arch = "64" 
    else:
        platform_name = "linux"
        arch = "aarch64" if "arm" in machine else "64"

    url = f"https://micro.mamba.pm/api/micromamba/{platform_name}-{arch}/latest"
    tar_path = os.path.join(MAMBA_ROOT, "mm.tar.bz2")
    
    print(f"    Target URL: {url}")

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
    
    subdir = "osx-64" if system == "darwin" else "linux-64"

    print(f"\n [1/2] Creating isolated lightweight environment via Micromamba...")
    cmd = [
        mamba_exe, "create", 
        "-p", ENV_PREFIX, 
        "-r", MAMBA_ROOT,
        "-c", "conda-forge", "-c", "igwn",
        "python=3.10", "gwpy", "hveto", "pandas", 
        "pillow", "tensorflow", "pytorch", "torchvision", 
        "--yes",
        "--platform", subdir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n Error Occured: Installation(Exit code: {e.returncode})")
        sys.exit(1)

    print(f"\n [2/2] Generating activation script...")
    activate_script = os.path.join(BASE_DIR, "activate_igwn_env.sh")
    with open(activate_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export CONDA_SUBDIR={subdir}\n")
        f.write(f'SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n')
        f.write(f'eval "$("$SCRIPT_DIR/.micromamba/micromamba" shell hook --shell bash)"\n')
        f.write(f'micromamba activate "$SCRIPT_DIR/python_env"\n')
        f.write('echo "Local Lightweight IGWN & ML Environment Activated!"\n')
    
    os.chmod(activate_script, 0o755)
    
    print(f"\n Setup Complete! Command: source ./activate_igwn_env.sh")

if __name__ == "__main__":
    main()
