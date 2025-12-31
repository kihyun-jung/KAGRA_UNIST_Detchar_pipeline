#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coherence Job Setup
===================

HTCondor Submission 파일(.sub)을 자동으로 생성합니다.
1. Overall Coherence용 작업 파일
2. Glitch Coherence용 작업 파일
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# KISTI Conda 환경 설정 (하드코딩 유지하되 변수화)
CONDA_EXEC = "/cvmfs/software.igwn.org/conda/bin/conda"
CONDA_ENV = "/cvmfs/software.igwn.org/conda/envs/igwn-py36-20210512"

def create_submission_file(job_type, script_name):
    """
    job_type: 'overall' or 'glitch'
    script_name: 실행할 파이썬 스크립트 파일명
    """
    script_path = BASE_DIR / "src" / "analysis" / "coherence" / script_name
    channel_list = BASE_DIR / "results" / "coherence" / "channel_list.txt"
    log_dir = BASE_DIR / "logs"
    
    sub_filename = BASE_DIR / "jobs" / f"submit_{job_type}_coherence.sub"
    
    # 로그 폴더 생성
    log_dir.mkdir(exist_ok=True)
    (BASE_DIR / "jobs").mkdir(exist_ok=True)

    with open(sub_filename, 'w') as f:
        f.write("Universe = vanilla\n")
        f.write(f"Executable = {CONDA_EXEC}\n")
        
        # Conda Run을 사용하여 환경 로드 후 스크립트 실행
        args = f"run -p {CONDA_ENV} --no-capture-output python3 {script_path} -ch $(CHANNEL)"
        f.write(f"Arguments = {args}\n")
        
        f.write("request_cpus = 4\n")
        f.write("request_memory = 8192\n")
        f.write("getenv = True\n")
        
        f.write(f"Log = {log_dir}/{job_type}_$(CHANNEL).log\n")
        f.write(f"Output = {log_dir}/{job_type}_$(CHANNEL).out\n")
        f.write(f"Error = {log_dir}/{job_type}_$(CHANNEL).err\n")
        
        f.write("notification = never\n")
        
        # Queue 명령어: 채널 리스트 파일에서 한 줄씩 읽어서 $(CHANNEL) 변수에 대입
        f.write(f"queue CHANNEL from {channel_list}\n")

    print(f"[*] Created submission file: {sub_filename}")

if __name__ == "__main__":
    create_submission_file("overall", "calc_overall.py")
    create_submission_file("glitch", "calc_glitch.py")
