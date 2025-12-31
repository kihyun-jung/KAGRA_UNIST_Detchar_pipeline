#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hveto Setup & Configuration (Production)
========================================

Omicron Trigger 결과를 바탕으로 Hveto 분석 환경을 구축합니다.
KISTI CVMFS 환경(igwn-py36-20210512)의 실제 Hveto를 호출하도록 설정합니다.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

# 모듈 Import를 위한 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.etl.generate_ffl import generate_ffl

TEMPLATE_PATH = BASE_DIR / "config" / "templates" / "hveto_config_template.ini"

# ==========================================
# Environment Configuration (KISTI Cluster)
# ==========================================
CONDA_EXEC = "/cvmfs/software.igwn.org/conda/bin/conda"
CONDA_ENV = "/cvmfs/software.igwn.org/conda/envs/igwn-py36-20210512"
HVETO_BIN = "/data/kagra/home/kagradet/.local/bin/hveto-kisti" # 혹은 해당 환경 내부의 hveto

def setup_hveto_for_date(year, month, day):
    target_date = datetime.date(year, month, day)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"[*] Setting up Hveto (Production Mode) for {date_str}...")

    # 1. 출력 디렉토리 생성
    output_dir = BASE_DIR / "results" / "hveto" / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # 2. FFL 생성 (Omicron 결과 연결)
    # 실제 데이터 범위를 커버하도록 GPS 설정 (데모용 값 유지 또는 실제 값으로 변경 필요)
    search_start = 1371090000 
    search_end = 1372000000
    trigger_pattern = str(BASE_DIR / "results" / "omicron" / date_str / "triggers" / "*.xml")
    
    main_ffl = output_dir / "K1-main.ffl"
    aux_ffl = output_dir / "K1-aux.ffl"

    count = generate_ffl(search_start, search_end, trigger_pattern, str(main_ffl))
    generate_ffl(search_start, search_end, trigger_pattern, str(aux_ffl))
    
    # 3. Config (INI) 생성
    if not TEMPLATE_PATH.exists():
        print(f"[Error] Template not found: {TEMPLATE_PATH}")
        return False

    with open(TEMPLATE_PATH, 'r') as f:
        config_content = f.read()
    
    # 실제 환경이므로 Mock Channel 대신 실제 보조 채널을 넣거나,
    # 보안상 비워두더라도 코드는 '실제 실행'을 가정하고 작성됨
    # 여기서는 GitHub 공개용이므로 여전히 Mock 또는 Blank 처리를 권장하나,
    # 로직상 {AUX_CHANNELS} 부분은 채워져야 에러가 안 납니다.
    mock_channels = "K1:PEM-DEMO_SENSOR_FOR_GITHUB" 
    config_content = config_content.replace("{AUX_CHANNELS}", mock_channels)
    
    ini_path = output_dir / "hveto.ini"
    with open(ini_path, 'w') as f:
        f.write(config_content)

    # 4. Condor Submission 파일 생성 (Real Environment)
    sub_path = output_dir / "submit_hveto.sub"
    
    # 세그먼트 파일 경로 (이전 단계에서 생성된 것)
    segment_file = BASE_DIR / "segments_output.txt" 
    # (주의: 실제 hveto는 xml 세그먼트를 원할 수 있으므로 확인 필요, 여기선 txt 가정)

    with open(sub_path, 'w') as f:
        f.write("universe = vanilla\n")
        
        # [핵심] Conda Run을 통해 특정 환경의 python/hveto 실행
        f.write(f"executable = {CONDA_EXEC}\n")
        
        # Arguments: run -p {ENV} python {HVETO} ...
        # 주의: 경로에 공백이 없다고 가정. 절대 경로 사용 필수.
        hveto_cmd = (
            f"run -p {CONDA_ENV} --no-capture-output "
            f"python3 {HVETO_BIN} "
            f"{search_start} {search_end} "
            f"--ifo K1 "
            f"--config-file hveto.ini "
            f"--primary-cache K1-main.ffl "
            f"--auxiliary-cache K1-aux.ffl "
            f"--analysis-segments {segment_file} "
            f"--omega-scans 5"
        )
        
        f.write(f"arguments = {hveto_cmd}\n")
        
        f.write("environment = KMP_LIBRARY=serial;MKL_SERIAL=yes\n")
        f.write("request_cpus = 8\n")
        f.write("request_memory = 8192\n")
        f.write("getenv = True\n")
        
        # 로그 파일
        f.write(f"output = {output_dir}/logs/hveto.out\n")
        f.write(f"error = {output_dir}/logs/hveto.err\n")
        f.write(f"log = {output_dir}/logs/hveto.log\n")
        
        # 파일 전송 (필요한 설정 파일들)
        # transfer_input_files = hveto.ini, K1-main.ffl, ... (경로 맞춰야 함)
        
        f.write("queue 1\n")

    print(f"[*] Production Hveto setup complete for {date_str}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    setup_hveto_for_date(args.year, args.month, args.day)
