#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# ==========================================
# Path Configuration (Updated)
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def check_environment():
    """로컬 환경 내에 omicron 설치 확인"""
    if not shutil.which("omicron"):
        print("[!] Error: 'omicron' command not exist. Did you activate environment?")
        sys.exit(1)

def read_gps_from_segments(seg_file):
    """세그먼트 파일에서 GPS Start와 End 시간을 읽어옴"""
    try:
        with open(seg_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            if not lines:
                raise ValueError("Empty segments.")
            parts = lines[0].split()
            return parts[0], parts[1]
    except Exception as e:
        print(f"[!] Segments file Error: {e}")
        sys.exit(1)

def generate_ffl(raw_dir, date_str, output_dir):
    """
    KAGRA 표준 FFL 형식에 맞춰 파일 생성
    형식: [Full Path] [GPS Start] [Duration] 0 0
    """
    ffl_path = output_dir / f"cache_{date_str}_mock.ffl"
    gwf_files = sorted(list(raw_dir.glob("*.gwf")))
    
    if not gwf_files:
        print(f"[!] No GWF files: {raw_dir}.")
        return None

    with open(ffl_path, "w") as f:
        for gwf in gwf_files:
            # 파일명 형식 예: K-K1_C-1371081600-32.gwf
            parts = gwf.stem.split("-")
            try:
                t_start = parts[-2]
                dur = parts[-1]
                # Cache format: [Path] [Start] [Duration] 0 0
                f.write(f"{gwf.resolve()} {t_start} {dur} 0 0\n")
            except (IndexError, ValueError):
                print(f"[!] Parsing failed: {gwf.name}")
                continue
    return ffl_path

def generate_omicron_parameter(output_dir, channels, freq, ffl_path):
    """파라미터 파일 생성"""
    param_file = output_dir / f"parameter_mock_{freq}.txt"
    f_max = min(4096, int(freq * 0.4)) 
    
    content = f"""// Omicron configuration
PARAMETER TIMING 64 4
PARAMETER FREQUENCYRANGE 10 {f_max}
PARAMETER QRANGE 4 128
PARAMETER MISMATCHMAX 0.2
PARAMETER SNRTHRESHOLD 6
PARAMETER PSDLENGTH 128
PARAMETER CLUSTERING TIME
PARAMETER CLUSTERDT 0.1

//** output configuration
OUTPUT DIRECTORY {output_dir.resolve()}
OUTPUT PRODUCTS triggers
OUTPUT FORMAT xml
OUTPUT VERBOSITY 0

//** data configuration
DATA FFL {ffl_path.resolve()}
DATA SAMPLEFREQUENCY {freq}
PARAMETER TRIGGERRATEMAX 10000

DATA CHANNELS {" ".join(channels)}
"""
    with open(param_file, "w") as f:
        f.write(content)
    return param_file

def main():
    parser = argparse.ArgumentParser(description="KAGRA Omicron Pipeline")
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    parser.add_argument("--mode", choices=['mock', 'real'], default='mock')
    args = parser.parse_args()

    check_environment()
    pure_date = f"{args.year}-{args.month:02d}-{args.day:02d}"
    
    # RESULTS_DIR 기반 경로 설정
    base_path = RESULTS_DIR / f"{pure_date}_{args.mode}"
    raw_dir = base_path / "raw"
    omicron_out_dir = base_path / "omicron"
    omicron_out_dir.mkdir(parents=True, exist_ok=True)
    
    seg_file = omicron_out_dir / f"segments_{pure_date}_{args.mode}.txt"
    gps_start, gps_end = read_gps_from_segments(seg_file)

    # 2. FFL 생성
    ffl_path = generate_ffl(raw_dir, pure_date, omicron_out_dir)
    if not ffl_path: return

    # 3. 채널 분류
    freq_map = {16384: ["K1:CAL-MOCK"]}
    for f in [512, 1024, 2048, 4096, 8192, 16384]:
        freq_map.setdefault(f, []).extend([f"K1:AUX-CHANNEL_{f}_1_DQ", f"K1:AUX-CHANNEL_{f}_2_DQ"])

    # 4. 파라미터 생성
    param_tasks = []
    for freq, channels in freq_map.items():
        p_file = generate_omicron_parameter(omicron_out_dir, channels, freq, ffl_path)
        param_tasks.append((freq, p_file))

    # 5. 실행
    if args.mode == 'mock':
        print(f"[*] Mock mode: Omicron start (GPS: {gps_start}-{gps_end})")
        for freq, p_file in param_tasks:
            print(f"    - {freq} Hz making...")
            cmd = ["omicron", str(gps_start), str(gps_end), str(p_file.resolve())]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"    [!] {freq} Hz failed.")
        print("\n All Mock data is refined.")

if __name__ == "__main__":
    main()
