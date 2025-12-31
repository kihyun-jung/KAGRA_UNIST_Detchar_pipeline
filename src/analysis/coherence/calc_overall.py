#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overall Coherence Calculator
============================

Hveto 분석에서 식별된 채널 중 '가장 심각한 날짜(Max Significance Date)'를 선정하여,
하루 동안의 장기적인 주파수 상관관계(Coherence)를 계산합니다.
"""

import sys
import os
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # 서버 환경용
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# GWpy
try:
    from gwpy.timeseries import TimeSeries
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.time import tconvert # KISTI tconvert 대체
except ImportError:
    print("[!] 'gwpy' not installed.")
    sys.exit(1)

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# --- Configuration ---
MAIN_CHANNEL = "K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ"
GWF_DURATION = 32.0
FFT_LENGTH = 2.0
OVERLAP = 0.5

# 입력 데이터 경로 (Hveto 결과 파싱된 파일들)
EXTRACTED_DIR = BASE_DIR / "results" / "coherence" / "extracted_channels"

def get_max_sig_date(channel_name):
    """
    채널별 분석 파일(.txt)을 읽어 가장 Significance가 높은 날짜를 반환합니다.
    """
    file_path = EXTRACTED_DIR / f"{channel_name}.txt"
    if not file_path.exists():
        print(f"[!] Channel file not found: {file_path}")
        return None

    max_sig = -1.0
    best_date_str = None
    year = 2023 # O4a Year

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            try:
                # Format: MM-DD Round Sig
                mm_dd = parts[0]
                sig = float(parts[2])
                
                if sig > max_sig:
                    max_sig = sig
                    best_date_str = f"{year}-{mm_dd}"
            except ValueError:
                continue
    
    if best_date_str:
        print(f"[*] Best Date for {channel_name}: {best_date_str} (Sig: {max_sig})")
    return best_date_str

def calculate_overall_coherence(channel_name):
    target_date = get_max_sig_date(channel_name)
    if not target_date:
        return

    # FFL 경로 (Omicron 단계에서 생성된 FFL 사용)
    ffl_path = BASE_DIR / f"kagra_data_{target_date}.ffl"
    if not ffl_path.exists():
        # 데모용: 없으면 가장 최근 FFL 찾기 시도하거나 에러
        print(f"[!] FFL not found for {target_date}. Checking demo files...")
        # (Demo logic omitted for brevity, exiting)
        return

    print(f"[*] Calculating Coherence: Main vs {channel_name} on {target_date}")
    
    # --- (중략) 실제 계산 로직은 원본의 핵심 흐름을 유지하되 ---
    # --- gwpy.timeseries.read() 부분에서 ffl_path를 사용하도록 수정됨 ---
    
    # [Mock Implementation for Portfolio Showcase]
    # 실제 데이터가 없으므로 Random Noise로 계산 과정을 시연합니다.
    print("[*] Generating Mock Data for demonstration...")
    duration = 600 # 10분 데이터
    fs = 2048
    t0 = tconvert(target_date)
    
    main_data = TimeSeries(np.random.randn(duration*fs), sample_rate=fs, t0=t0, name=MAIN_CHANNEL)
    aux_data = TimeSeries(np.random.randn(duration*fs), sample_rate=fs, t0=t0, name=channel_name)
    
    # Coherence 계산
    coh = main_data.coherence(aux_data, fftlength=FFT_LENGTH, overlap=OVERLAP)
    
    # Plotting
    plot = coh.plot(figsize=(12, 6))
    ax = plot.gca()
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Coherence')
    ax.set_title(f"Overall Coherence ({target_date})\n{MAIN_CHANNEL} vs {channel_name}")
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # 저장
    output_dir = BASE_DIR / "results" / "coherence" / "overall_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_ch = channel_name.replace(":", "_")
    save_path = output_dir / f"Overall_{safe_ch}_{target_date}.png"
    plot.savefig(save_path)
    print(f"✅ Saved plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ch", "--channel", required=True, help="Target Channel Name")
    args = parser.parse_args()
    
    calculate_overall_coherence(args.channel)
