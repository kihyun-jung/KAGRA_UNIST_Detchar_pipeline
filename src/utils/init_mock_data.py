#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock Data Initializer
=====================

이 스크립트는 GitHub 포트폴리오용 데모를 위해
세그먼트 예시 파일(1371xxxxxx)과 시간이 일치하는
Dummy Raw Data(GWF) 파일 20개를 생성합니다.

생성되는 데이터 구간:
Start GPS: 1371097726 (2023-06-19 경)
Duration per file: 32 seconds
Total files: 20
"""

import os
from pathlib import Path

# 프로젝트 루트 및 데이터 저장 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw_sample"

def create_mock_files():
    # 1. 디렉토리 생성 (없으면 생성)
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created directory: {RAW_DATA_DIR}")

    # 2. 파일 생성 설정
    # 아까 주신 세그먼트 예시 중 두 번째 줄(1371097726)부터 시작
    start_gps = 1371097726 
    duration = 32 # 일반적인 Raw Data 청크 단위
    num_files = 20

    print(f"[*] Generating {num_files} mock GWF files...")

    # 3. 파일 생성 루프
    for i in range(num_files):
        current_gps = start_gps + (i * duration)
        
        # 파일명 형식: K-K1_C-{GPS}-{DURATION}.gwf
        # KAGRA 표준 명명 규칙을 따름
        filename = f"K-K1_C-{current_gps}-{duration}.gwf"
        filepath = RAW_DATA_DIR / filename
        
        # 0바이트 파일 생성 (내용은 중요하지 않음, 파일명이 중요)
        with open(filepath, 'w') as f:
            pass # 빈 파일 생성
            
        print(f"  - Created: {filename}")

    print("="*50)
    print(f"[*] Done! {num_files} files are ready in 'data/raw_sample/'.")
    print(f"[*] Time range: {start_gps} ~ {start_gps + (num_files * duration)}")

if __name__ == "__main__":
    create_mock_files()
