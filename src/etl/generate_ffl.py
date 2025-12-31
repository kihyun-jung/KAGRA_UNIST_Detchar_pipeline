#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAGRA Frame File List (FFL) Generator
=====================================
(Updated for Universal Usage: GWF & XML)
"""

import os
import glob
import argparse
import sys

def parse_filename(filepath):
    """
    파일명에서 GPS 시작 시간과 길이(Duration)를 추출합니다.
    지원 포맷:
      1. GWF: Source-Type-GPS-DURATION.gwf
      2. XML: Source-Name-GPS-DURATION.xml
    """
    try:
        filename = os.path.basename(filepath)
        # 확장자 제거 ('file.xml.gz' 같은 경우도 고려하여 splitext 대신 처리)
        name_body = filename.split('.')[0] 
        
        # '-' 기준으로 분리
        parts = name_body.split('-')
        
        # LIGO/KAGRA 표준: 마지막 두 부분이 GPS Start와 Duration임
        if len(parts) >= 2:
            gps_start = float(parts[-2])
            duration = float(parts[-1])
            return gps_start, duration
        else:
            return None, None
            
    except (IndexError, ValueError):
        return None, None

def generate_ffl(start_time, end_time, patterns, output_file):
    """
    주어진 패턴의 파일 중 시간 범위가 겹치는 파일만 필터링하여 FFL을 작성합니다.
    (다른 스크립트에서 import해서 쓸 수 있도록 설계됨)
    """
    file_list = []
    
    # patterns가 리스트인지 문자열인지 확인하여 처리
    if isinstance(patterns, list):
        glob_patterns = patterns
    else:
        glob_patterns = patterns.strip().split(",")
    
    for pattern in glob_patterns:
        found_files = glob.glob(pattern)
        file_list.extend(found_files)
    
    file_list = sorted(file_list)
    
    # print(f"[*] Found {len(file_list)} files. Filtering...") # 너무 시끄러우면 주석 처리

    selected_files = []

    for f_path in file_list:
        f_start, f_duration = parse_filename(f_path)
        
        if f_start is None:
            continue 

        f_end = f_start + f_duration
        overlap = max(f_start, start_time) < min(f_end, end_time)

        if overlap:
            selected_files.append((f_path, int(f_start), int(f_duration)))

    # FFL 파일 작성
    try:
        with open(output_file, 'w') as out:
            for path, gps, dur in selected_files:
                abs_path = os.path.abspath(path)
                # FFL 표준 포맷: 경로 GPS Dur 0 0
                out.write(f"{abs_path} {gps} {dur} 0 0\n")
        
        # 호출한 쪽에서 몇 개가 써졌는지 알 수 있게 리턴
        return len(selected_files)

    except IOError as e:
        print(f"[Error] Failed to write FFL: {e}")
        return 0

# 메인 실행부 (CLI 용도)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start-gpstime', type=float, required=True)
    parser.add_argument('-e', '--end-gpstime', type=float, required=True)
    parser.add_argument('-o', '--output-filename', type=str, required=True)
    parser.add_argument('-p', '--globbing-pattern', type=str, required=True)

    args = parser.parse_args()

    count = generate_ffl(args.start_gpstime, args.end_gpstime, args.globbing_pattern, args.output_filename)
    print(f"[*] Generated FFL with {count} entries.")
