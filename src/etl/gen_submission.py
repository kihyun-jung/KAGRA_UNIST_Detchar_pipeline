#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTCondor Submission Generator for Omicron
=========================================

이 스크립트는 KISTI 슈퍼컴퓨팅 환경(HTCondor)에 작업을 제출하기 위한 
.sub 파일을 생성합니다. (Target: 16384Hz)
"""

import os
import argparse
from pathlib import Path

# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent
JOBS_DIR = BASE_DIR / "jobs"
LOGS_DIR = BASE_DIR / "logs"

# 실행 파일 경로 (환경에 따라 수정 필요, 여기서는 예시 경로 사용)
OMICRON_BIN = "/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py38-20220827/bin/omicron"

def generate_submission(year, month, day):
    # 디렉토리 생성
    if not JOBS_DIR.exists(): JOBS_DIR.mkdir()
    if not LOGS_DIR.exists(): LOGS_DIR.mkdir()

    # 날짜 문자열
    date_str = f"{year}-{month:02d}-{day:02d}"

    # 파라미터 파일 경로 (1번 스크립트에서 생성된 파일)
    param_file = f"../config/parameters_kagra_16384_{date_str}.txt"
    segment_file = "../segments_output.txt" # 이전 단계에서 생성된 세그먼트 파일

    # 서브미션 파일명
    sub_filename = JOBS_DIR / f"omicron_16384_{date_str}.sub"

    print(f"[*] Generating submission file: {sub_filename}")

    with open(sub_filename, 'w') as f:
        f.write('universe = vanilla\n')
        f.write(f'executable = {OMICRON_BIN}\n')
        
        # 상대 경로 사용
        f.write(f'arguments = {segment_file} {param_file}\n')
        
        f.write('environment = "KMP_LIBRARY=serial;MKL_SERIAL=yes"\n')
        f.write('request_cpus = 4\n')
        f.write('request_memory = 8192\n') # 4 * 2048
        
        # [Security] 특정 노드 제외 옵션은 일반화하거나 주석 처리함
        # f.write('requirements = (Machine != "specific-node-01") ... \n')
        
        f.write('getenv = True\n')
        
        # 로그 파일 경로
        f.write(f'log = {LOGS_DIR}/omicron_16384.log\n')
        f.write(f'error = {LOGS_DIR}/omicron_16384.err\n')
        f.write(f'output = {LOGS_DIR}/omicron_16384.out\n')
        
        f.write('notification = never\n')
        f.write('queue 1\n')

    print("[*] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    generate_submission(args.year, args.month, args.day)
