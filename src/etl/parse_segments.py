#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAGRA Segment Parser for Omicron Pipeline
=========================================

이 스크립트는 KAGRA 검출기의 상태(Science Mode 등)를 기록한 세그먼트 파일을 읽어,
Omicron 분석 파이프라인에서 사용할 수 있는 표준 포맷으로 변환합니다.

Usage:
    python parse_segments.py -y 2023 -m 6 -d 13
"""

import os
import argparse
import datetime
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
# 프로젝트 루트 디렉토리 설정 (현재 스크립트 위치 기준 상위 2단계)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "segments"

def get_gps_time(date_str):
    """
    날짜 문자열(YYYY-MM-DD)을 GPS 시간으로 변환합니다.
    (실제 환경에서는 'tconvert' 명령어를 사용하지만, 데모를 위해 가상의 변환 로직 사용)
    """
    try:
        # 실제 환경: int(os.popen(f"tconvert {date_str}").read())
        # 데모 환경: 1980년 1월 6일(GPS Epoch) 기준 차이를 계산 (Approx.)
        epoch = datetime.datetime(1980, 1, 6)
        target = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        delta = target - epoch
        return int(delta.total_seconds())
    except Exception as e:
        print(f"[Warning] Time conversion failed: {e}")
        return 0

def parse_segment_file(year, month, day):
    """
    지정된 날짜의 세그먼트 파일을 읽어 유효한 세그먼트 리스트를 생성합니다.
    """
    # 1. 날짜 포맷팅 (YYYY-MM-DD) - datetime 모듈 활용
    target_date = datetime.date(year, month, day)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"[*] Processing date: {date_str}")

    # 2. 파일 경로 설정 (상대 경로 활용)
    # 실제 파일명 포맷에 맞춤
    filename = f"SCIENCE_MODE_EXAMPLE_SEGMENT_UTC_{date_str}.txt"
    input_path = DATA_DIR / filename
    output_path = BASE_DIR / "segments_output.txt"

    # 3. 파일 존재 여부 확인
    if not input_path.exists():
        print(f"[Error] Segment file not found: {input_path}")
        print("Please check if the file exists in 'data/segments/' directory.")
        return

    # 4. 세그먼트 파싱 및 저장
    valid_segments = []
    total_duration = 0

    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                start = int(parts[0])
                end = int(parts[1])
                duration = end - start
                
                # 유효성 검사 (음수 구간 등 방지)
                if duration > 0:
                    valid_segments.append((start, end))
                    total_duration += duration

        # 결과 저장
        with open(output_path, 'w') as g:
            for start, end in valid_segments:
                g.write(f"{start} {end}\n")
        
        print(f"[*] Successfully parsed {len(valid_segments)} segments.")
        print(f"[*] Total duration: {total_duration} seconds")
        print(f"[*] Output saved to: {output_path}")

    except Exception as e:
        print(f"[Error] Failed to parse segment file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse KAGRA segment files for Omicron analysis.")
    parser.add_argument("-y", "--year", required=True, type=int, help="Year (e.g., 2023)")
    parser.add_argument("-m", "--month", required=True, type=int, help="Month (e.g., 6)")
    parser.add_argument("-d", "--day", required=True, type=int, help="Day (e.g., 13)")
    
    args = parser.parse_args()
    
    parse_segment_file(args.year, args.month, args.day)
