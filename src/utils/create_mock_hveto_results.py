#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock Hveto Result Generator
===========================

Hveto 분석이 완료된 것처럼 가장하여,
1. hveto.out (로그 파일)
2. K1-HVETO_VETOED_TRIGS_ROUND_*.txt (Veto된 트리거 파일)
을 생성합니다.

보안상 민감한 실제 채널명 대신 Mock Channel 이름을 사용합니다.
"""

import os
from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def create_hveto_log(date_str, output_dir):
    """
    실제 hveto.out 형식을 흉내낸 로그 파일을 생성합니다.
    (채널명은 Mock으로 대체됨)
    """
    log_content = f"""hveto-kisti 2023-06-18 14:59:47 KST     INFO: -- Welcome to Hveto --
hveto-kisti 2023-06-18 14:59:47 KST     INFO: GPS start time: 1371081618
hveto-kisti 2023-06-18 14:59:47 KST     INFO: GPS end time: 1371168018
hveto-kisti 2023-06-18 14:59:47 KST     INFO: Interferometer: K1
hveto-kisti 2023-06-18 14:59:47 KST     INFO: Working directory: {output_dir}
hveto-kisti 2023-06-18 14:59:54 KST     INFO: Identified 593 auxiliary channels to process

hveto-kisti 2023-06-18 20:54:58 KST     INFO: -- Processing round 1 --
hveto-kisti 2023-06-18 20:55:20 KST     INFO: Round 1 winner: K1:PEM-MOCK_SENSOR_1
hveto-kisti 2023-06-18 20:55:36 KST     INFO: Results for round 1:

winner :          K1:PEM-MOCK_SENSOR_1
significance :    15.568641543966532
mu :              21.931940337521034
snr :             40.0
dt :              0.4
use_percentage :  (70, 534)
efficiency :      (70, 6285)
deadtime :        (213.26255083084106, 61211.0)
cum. efficiency : (70, 6285)
cum. deadtime :   (213.26255083084106, 61211.0)

hveto-kisti 2023-06-18 20:55:42 KST     INFO: -- Processing round 2 --
hveto-kisti 2023-06-18 20:56:02 KST     INFO: Round 2 winner: K1:PEM-MOCK_SENSOR_2
hveto-kisti 2023-06-18 20:56:24 KST     INFO: Results for round 2:

winner :          K1:PEM-MOCK_SENSOR_2
significance :    15.182135136812866
mu :              2.526847821666202
snr :             8.0
dt :              0.4
use_percentage :  (24, 62)

hveto-kisti 2023-06-18 20:56:27 KST     INFO: -- Processing round 3 --
hveto-kisti 2023-06-18 20:56:48 KST     INFO: Round 3 winner: K1:PEM-MOCK_SENSOR_3
hveto-kisti 2023-06-18 20:57:09 KST     INFO: Results for round 3:

winner :          K1:PEM-MOCK_SENSOR_3
significance :    10.9582
mu :              9.0367
snr :             7.75

hveto-kisti 2023-06-18 20:57:11 KST     INFO: -- Processing round 4 --
hveto-kisti 2023-06-18 20:57:31 KST     INFO: Round 4 winner: K1:PEM-MOCK_SENSOR_4
hveto-kisti 2023-06-18 20:57:52 KST     INFO: Results for round 4:

winner :          K1:PEM-MOCK_SENSOR_4
significance :    6.6967
mu :              41.8512
snr :             20.0

hveto-kisti 2023-06-18 20:57:55 KST     INFO: -- Processing round 5 --
hveto-kisti 2023-06-18 20:58:15 KST     INFO: Round 5 winner: K1:PEM-MOCK_SENSOR_5
hveto-kisti 2023-06-18 20:58:36 KST     INFO: Results for round 5:

winner :          K1:PEM-MOCK_SENSOR_5
significance :    5.8884
mu :              10.8511
snr :             100.0

hveto-kisti 2023-06-18 20:59:39 KST     INFO: -- Hveto complete --
"""
    # 로그 폴더 생성 및 파일 쓰기
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / "hveto.out", "w") as f:
        f.write(log_content)
    print(f"[*] Created mock log: {log_dir / 'hveto.out'}")

def create_vetoed_triggers(date_str, output_dir):
    """
    각 라운드별로 Veto된 Trigger 목록 파일(.txt)을 생성합니다.
    실제 데이터 시간을 기반으로 Mock 데이터를 채웁니다.
    """
    trigger_dir = output_dir / "triggers"
    trigger_dir.mkdir(parents=True, exist_ok=True)

    # 1371097728 주변 시간대 (오미크론 결과와 매칭)
    base_gps = 1371097728.0 
    main_ch = "K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ"
    
    # 5개 라운드에 대해 파일 생성
    for i in range(1, 6):
        filename = f"K1-HVETO_VETOED_TRIGS_ROUND_{i}-1371081618-86400.txt"
        filepath = trigger_dir / filename
        
        with open(filepath, "w") as f:
            # Header
            f.write("time peak_frequency snr channel\n")
            
            # Mock Data Rows (각 라운드마다 시간 조금씩 다르게)
            # 파싱 코드 테스트를 위해 SNR > 8.0과 < 8.0을 섞어서 생성
            f.write(f"{base_gps + i*10 + 0.5} 371.88 9.38 {main_ch}\n") # Over 8
            f.write(f"{base_gps + i*10 + 2.0} 364.10 7.50 {main_ch}\n") # Under 8
            f.write(f"{base_gps + i*10 + 5.5} 702.53 15.2 {main_ch}\n") # Over 8
            f.write(f"{base_gps + i*10 + 9.1} 590.60 8.01 {main_ch}\n") # Over 8
            
        print(f"[*] Created mock trigger file: {filepath}")

def main():
    # 데모 날짜: 2023-06-18
    date_str = "2023-06-18"
    output_dir = BASE_DIR / "results" / "hveto" / date_str
    
    print(f"🚀 Generating Mock Hveto Results for {date_str}...")
    
    create_hveto_log(date_str, output_dir)
    create_vetoed_triggers(date_str, output_dir)
    
    print("✅ All mock files generated successfully.")
    print("   Now you can run 'src/analysis/parse_hveto_results.py' to generate Q-scans.")

if __name__ == "__main__":
    main()
