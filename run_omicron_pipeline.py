#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAGRA Omicron Pipeline Orchestrator
===================================

[Execution Flow]
1. Mock Data Generation (For Demo)
2. Segment Parsing
3. FFL Generation
4. Parameter Configuration
5. HTCondor Submission File Generation
6. Job Submission Simulation & Output Mocking (New!)

Usage:
    python run_omicron_pipeline.py -y 2023 -m 6 -d 19
    (XML 파일의 GPS 시간 1371097728은 2023-06-19 입니다)
"""

import sys
import os
import argparse
import subprocess
import shutil
import datetime
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src" / "etl"
UTILS_DIR = BASE_DIR / "src" / "utils"

# 사용자 제공 XML 샘플 파일 정보
SAMPLE_XML_NAME = "K1-CAL_CS_PROC_DARM_STRAIN_DBL_DQ_OMICRON-1371097728-60.xml"
SAMPLE_XML_PATH = BASE_DIR / "data" / "sample_output" / SAMPLE_XML_NAME

def run_step(step_name, command):
    print("=" * 60)
    print(f"[*] Step: {step_name}")
    try:
        subprocess.run(command, check=True, text=True)
        print(f"[*] {step_name} Completed Successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error in {step_name}: {e}")
        sys.exit(1)

def simulate_output_generation(date_str):
    """
    오미크론 바이너리가 없는 환경에서, 
    미리 준비된 XML 샘플 파일을 결과 폴더로 복사하여 
    마치 분석이 완료된 것처럼 시뮬레이션합니다.
    """
    print("=" * 60)
    print("[*] Step: Simulating Omicron Output Generation")
    
    # 결과가 저장될 경로 (config_omicron.py와 일치해야 함)
    # results/omicron/{YYYY-MM-DD}/triggers/
    target_dir = BASE_DIR / "results" / "omicron" / date_str / "triggers"
    
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created result directory: {target_dir}")
        
    target_file = target_dir / SAMPLE_XML_NAME
    
    if SAMPLE_XML_PATH.exists():
        shutil.copy(SAMPLE_XML_PATH, target_file)
        print(f"[*] Copying sample XML to: {target_file}")
        print("[*] ✅ Output Simulation SUCCESS: Next pipeline (Hveto) is ready to run.")
    else:
        print(f"[!] Warning: Sample XML not found at {SAMPLE_XML_PATH}")
        print("    Please save the XML content to 'data/sample_output/' folder.")

def main(year, month, day):
    target_date = datetime.date(year, month, day)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"🚀 Starting Omicron Pipeline for Date: {date_str}")
    
    # Step 0: Mock Data Generation
    run_step("Mock Data Generation", ["python", str(UTILS_DIR / "init_mock_data.py")])

    # Step 1: Segment Parsing
    run_step("Segment Parsing", 
             ["python", str(SRC_DIR / "parse_segments.py"),
              "-y", str(year), "-m", str(month), "-d", str(day)])

    # Step 2: FFL Generation
    # (XML 파일의 GPS 시간에 맞춰 범위를 조정)
    start_gps = 1371097728 - 100 
    end_gps = 1371097728 + 100
    ffl_name = f"kagra_data_{date_str}.ffl"
    raw_pattern = str(BASE_DIR / "data" / "raw_sample" / "*.gwf")

    run_step("FFL Generation", 
             ["python", str(SRC_DIR / "generate_ffl.py"),
              "-s", str(start_gps), "-e", str(end_gps),
              "-o", ffl_name, "-p", raw_pattern])

    # Step 3: Configuration
    run_step("Parameter Configuration", 
             ["python", str(SRC_DIR / "config_omicron.py"),
              "-y", str(year), "-m", str(month), "-d", str(day)])

    # Step 4: Submission Generation
    run_step("Submission File Generation", 
             ["python", str(SRC_DIR / "gen_submission.py"),
              "-y", str(year), "-m", str(month), "-d", str(day)])

    # Step 5: HTCondor Submission (Simulation)
    print("=" * 60)
    print("[*] Step: HTCondor Job Submission")
    if shutil.which("condor_submit"):
        # 실제 환경
        sub_file = BASE_DIR / "jobs" / f"omicron_16384_{date_str}.sub"
        subprocess.run(["condor_submit", str(sub_file)])
    else:
        # 데모 환경
        print("[!] 'condor_submit' not found. Skipping actual submission.")
    
    # Step 6: Mock Output (핵심!)
    simulate_output_generation(date_str)
    
    print("=" * 60)
    print("\n🎉 Omicron Pipeline Execution Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", required=True, type=int)
    parser.add_argument("-m", "--month", required=True, type=int)
    parser.add_argument("-d", "--day", required=True, type=int)
    args = parser.parse_args()
    
    main(args.year, args.month, args.day)
