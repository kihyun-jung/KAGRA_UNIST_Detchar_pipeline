#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coherence Analysis Pipeline Orchestrator
========================================

Hveto 결과를 바탕으로 Coherence 분석을 위한 전체 과정을 자동화합니다.

[Execution Flow]
1. Data Prep: Mock HTML 생성 -> 파싱 -> 채널 리스트 추출
2. Job Setup: HTCondor용 .sub 파일 자동 생성
3. Execution:
   - (Cluster) 'condor_submit'이 있으면 실제 작업 제출
   - (Local) 없으면 첫 번째 채널에 대해 로컬 계산 테스트 (Smoke Test)

Usage:
    python run_coherence_pipeline.py
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
UTILS = BASE_DIR / "src" / "utils"
ANALYSIS_COH = BASE_DIR / "src" / "analysis" / "coherence"
RESULTS_DIR = BASE_DIR / "results" / "coherence"
JOBS_DIR = BASE_DIR / "jobs"

def run_step(desc, cmd):
    print("=" * 60)
    print(f"[*] Step: {desc}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {desc} Completed.")
    except subprocess.CalledProcessError:
        print(f"[!] Failed: {desc}")
        sys.exit(1)

def main():
    # 데모 날짜 설정
    target_date = "2023-06-18"
    
    print(f"🚀 Starting Coherence Pipeline for {target_date}")
    
    # ---------------------------------------------------------
    # Phase 1: Data Preparation & Mining
    # ---------------------------------------------------------
    run_step("Generating Mock HTML", 
             ["python", str(UTILS / "create_mock_html.py")])
    
    run_step("Parsing Hveto Results",
             ["python", str(ANALYSIS_COH / "parse_hveto_html.py"),
              "-s", target_date, "-e", target_date])
    
    run_step("Generating Channel List",
             ["python", str(ANALYSIS_COH / "gen_channel_list.py"),
              "-o", "channel_list.txt"])

    # ---------------------------------------------------------
    # Phase 2: Job Configuration
    # ---------------------------------------------------------
    # setup_coherence_jobs.py 실행 (Overall & Glitch용 .sub 파일 생성)
    run_step("Generating Condor Submission Files",
             ["python", str(ANALYSIS_COH / "setup_coherence_jobs.py")])

    # ---------------------------------------------------------
    # Phase 3: Execution (Smart Dispatch)
    # ---------------------------------------------------------
    print("=" * 60)
    print("[*] Step: Job Execution / Simulation")

    # 채널 리스트 읽기 (테스트용)
    channel_list_path = RESULTS_DIR / "channel_list.txt"
    if not channel_list_path.exists():
        print("[!] Channel list not found.")
        sys.exit(1)
        
    with open(channel_list_path, 'r') as f:
        channels = [line.strip() for line in f if line.strip()]
    
    if not channels:
        print("[!] No channels found to analyze.")
        sys.exit(0)

    # condor_submit 명령어 존재 여부 확인
    if shutil.which("condor_submit"):
        # [실제 환경] 클러스터에 작업 제출
        print(f"[*] Cluster environment detected. Submitting {len(channels)} jobs...")
        
        # Overall Coherence 작업 제출
        subprocess.run(["condor_submit", str(JOBS_DIR / "submit_overall_coherence.sub")])
        # Glitch Coherence 작업 제출
        subprocess.run(["condor_submit", str(JOBS_DIR / "submit_glitch_coherence.sub")])
        
        print("[*] 🚀 All jobs submitted to HTCondor successfully.")
        
    else:
        # [로컬/데모 환경] 첫 번째 채널만 로컬에서 계산 (Smoke Test)
        target_channel = channels[0]
        print(f"[*] 'condor_submit' not found (Local/Demo Environment).")
        print(f"[*] Performing SMOKE TEST on a single channel: {target_channel}")
        print("-" * 60)
        
        # 1. Overall Coherence 로컬 실행
        print(f"   > Running Overall Coherence for {target_channel}...")
        subprocess.run([
            "python", str(ANALYSIS_COH / "calc_overall.py"),
            "-ch", target_channel
        ], check=True)
        
        # 2. Glitch Coherence 로컬 실행
        print(f"   > Running Glitch Coherence for {target_channel}...")
        subprocess.run([
            "python", str(ANALYSIS_COH / "calc_glitch.py"),
            "-ch", target_channel
        ], check=True)
        
        print("-" * 60)
        print("[*] ✅ Local Smoke Test PASSED.")
        print(f"[*] Check results in: {RESULTS_DIR}/overall_plots/ and glitch_plots/")

    print("=" * 60)
    print("🎉 Coherence Pipeline Finished.")

if __name__ == "__main__":
    main()
