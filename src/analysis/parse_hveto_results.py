#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hveto Result Parser & Q-scan Pipeline
=====================================

1. Hveto 로그(hveto.out)를 파싱하여 각 Round의 Winner Channel을 식별합니다.
2. Vetoed Trigger 목록을 읽어옵니다.
3. 각 Glitch 이벤트에 대해 Q-scan 이미지를 생성합니다.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

# 모듈 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.analysis.generate_qscan import generate_qscan

def parse_hveto_log(log_path):
    """
    hveto.out 로그 파일을 읽어, Round별 Winner Channel 목록을 추출합니다.
    Returns: ['Channel_Round1', 'Channel_Round2', ...]
    """
    winner_channels = []
    
    if not log_path.exists():
        print(f"[!] Log file not found: {log_path}")
        return []

    with open(log_path, 'r') as f:
        for line in f:
            if "winner :" in line:
                # 라인 포맷: "winner :          K1:ASC-POP_..."
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    channel = parts[-1].strip()
                    winner_channels.append(channel)
    
    return winner_channels

def process_veto_results(year, month, day):
    target_date = datetime.date(year, month, day)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"[*] Parsing Hveto results for {date_str}...")

    # 경로 설정
    result_base = BASE_DIR / "results" / "hveto" / date_str
    log_file = result_base / "logs" / "hveto.out"
    
    # 1. Winner Channel 파싱
    winners = parse_hveto_log(log_file)
    print(f"[*] Identified {len(winners)} rounds of veto analysis.")
    
    # 2. 이미지 저장 경로
    qscan_base = BASE_DIR / "results" / "qscan" / date_str
    
    total_generated = 0
    main_channel = "K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ"
    duration = 0.5 # Q-scan 지속 시간

    # 3. 각 Round별 결과 처리
    for i, winner_ch in enumerate(winners):
        round_num = i + 1
        # 파일명 패턴: K1-HVETO_VETOED_TRIGS_ROUND_{N}-{GPS}-86400.txt
        # (실제로는 GPS 시간이 파일명에 붙지만, 데모를 위해 glob으로 찾음)
        trig_files = list(result_base.glob(f"triggers/K1-HVETO_VETOED_TRIGS_ROUND_{round_num}-*.txt"))
        
        if not trig_files:
            continue
            
        trig_file = trig_files[0]
        
        with open(trig_file, 'r') as f:
            lines = f.readlines()
            # 헤더(첫 줄) 건너뜀
            for line in lines[1:]:
                parts = line.split()
                if len(parts) < 3: continue
                
                # 파일 포맷: time | peak_frequency | snr | channel
                gps_time = float(parts[0])
                snr = float(parts[2])
                
                # SNR 기준 폴더 분류 (8.0 기준)
                folder_name = "over8" if snr > 8.0 else "under8"
                
                # A. 메인 채널 Q-scan 생성
                main_img_path = qscan_base / folder_name / f"Main-Round{round_num}-{gps_time:.2f}.png"
                generate_qscan(main_channel, gps_time, duration, str(main_img_path), snr)
                
                # B. Winner(Aux) 채널 Q-scan 생성
                aux_img_path = qscan_base / folder_name / f"Aux-{winner_ch}-{gps_time:.2f}.png"
                generate_qscan(winner_ch, gps_time, duration, str(aux_img_path), snr)
                
                total_generated += 2

    print(f"[*] Total {total_generated} Q-scan images generated in {qscan_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    process_veto_results(args.year, args.month, args.day)
