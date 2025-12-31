#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Omicron Parameter Configurator
==============================

이 스크립트는 'config/templates/omicron_param_template.txt'를 읽어,
{OUTPUT_DIRECTORY} 와 {DATA_FFL} 부분을 실제 분석 환경에 맞게 치환한 뒤
최종 설정 파일을 생성합니다.
"""

import argparse
import datetime
from pathlib import Path

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = BASE_DIR / "config" / "templates" / "omicron_param_template.txt"
OUTPUT_CONFIG_DIR = BASE_DIR / "config"

def update_parameters(year, month, day):
    # 1. 날짜 포맷팅
    target_date = datetime.date(year, month, day)
    date_str = target_date.strftime("%Y-%m-%d")
    
    # 2. 경로 설정 (자동화)
    # FFL 파일 위치 (이전 단계에서 생성된 ffl 파일명)
    # 예: kagra_data_2023-06-13.ffl
    ffl_filename = f"kagra_data_{date_str}.ffl"
    path_ffl = BASE_DIR / ffl_filename
    
    # 결과 출력 디렉토리
    # 예: My-KAGRA-Pipeline/results/omicron/2023-06-13
    path_output = BASE_DIR / "results" / "omicron" / date_str
    
    print(f"[*] Configuring for Date: {date_str}")
    print(f"[*] Template: {TEMPLATE_PATH}")
    
    # 3. 템플릿 읽기
    if not TEMPLATE_PATH.exists():
        print(f"[Error] Template file not found: {TEMPLATE_PATH}")
        return

    with open(TEMPLATE_PATH, 'r') as f:
        template_content = f.read()

    # 4. 내용 치환 (String Replacement)
    # 템플릿 파일 안의 {OUTPUT_DIRECTORY}와 {DATA_FFL}을 실제 경로로 바꿈
    new_content = template_content.replace("{OUTPUT_DIRECTORY}", str(path_output))
    new_content = new_content.replace("{DATA_FFL}", str(path_ffl))

    # 5. 최종 설정 파일 저장
    output_filename = f"parameters_kagra_16384_{date_str}.txt"
    output_path = OUTPUT_CONFIG_DIR / output_filename
    
    # config 폴더가 없으면 생성
    if not OUTPUT_CONFIG_DIR.exists():
        OUTPUT_CONFIG_DIR.mkdir(parents=True)

    with open(output_path, 'w') as f:
        f.write(new_content)

    print(f"[*] Configuration saved to: {output_path}")
    print(f"[*] Ready to run Omicron!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    update_parameters(args.year, args.month, args.day)
