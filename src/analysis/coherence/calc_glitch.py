#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Glitch (SNR-Weighted) Coherence Calculator
==========================================

Hveto가 탐지한 Glitch 발생 시점들만 골라내어,
SNR을 가중치로 한 고해상도 Coherence를 계산합니다.
순간적인 노이즈의 상관관계를 파악하는 데 특화되어 있습니다.
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.time import tconvert
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EXTRACTED_DIR = BASE_DIR / "results" / "coherence" / "extracted_channels"
MAIN_CHANNEL = "K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ"

def calc_glitch_coherence(channel_name):
    # 1. 트리거 파일 찾기 (데모용: extracted_channels 파일 사용)
    info_file = EXTRACTED_DIR / f"{channel_name}.txt"
    if not info_file.exists():
        print(f"[!] No trigger info for {channel_name}")
        return

    print(f"[*] Calculating SNR-Weighted Coherence for {channel_name}...")
    
    # [Mock Simulation Logic]
    # 포트폴리오 데모를 위해 가상의 Glitch Event 10개를 생성하여 계산합니다.
    
    n_events = 10
    fft_len = 0.5
    fs = 4096
    
    weighted_pxx = None
    weighted_pyy = None
    weighted_pxy = None
    total_snr = 0.0
    
    for i in range(n_events):
        snr = 10.0 + np.random.random() * 50 # Random SNR (10~60)
        
        # 가짜 Glitch 데이터 생성 (짧은 구간)
        noise = np.random.randn(int(2*fs)) 
        main_seg = TimeSeries(noise, sample_rate=fs)
        aux_seg = TimeSeries(noise + np.random.randn(int(2*fs))*0.5, sample_rate=fs) # 약간의 상관관계 부여
        
        # PSD/CSD 계산
        # [수정됨] method='welch' 제거 (gwpy .csd()는 기본적으로 welch 방식을 사용함)
        pxx = main_seg.psd(fftlength=fft_len, window='hann')
        pyy = aux_seg.psd(fftlength=fft_len, window='hann')
        pxy = main_seg.csd(aux_seg, fftlength=fft_len, window='hann')
        
        # SNR 가중 누적
        if weighted_pxx is None:
            weighted_pxx = pxx * snr
            weighted_pyy = pyy * snr
            weighted_pxy = pxy * snr
        else:
            weighted_pxx += pxx * snr
            weighted_pyy += pyy * snr
            weighted_pxy += pxy * snr
            
        total_snr += snr

    # 최종 Coherence 계산
    avg_pxx = weighted_pxx / total_snr
    avg_pyy = weighted_pyy / total_snr
    avg_pxy = weighted_pxy / total_snr
    
    coherence = (np.abs(avg_pxy)**2) / (avg_pxx * avg_pyy)
    
    # Plotting
    plot = coherence.plot(figsize=(12, 6))
    ax = plot.gca()
    ax.set_ylabel('Coherence (SNR Weighted)')
    ax.set_title(f"Glitch Coherence: {channel_name}")
    ax.set_ylim(0, 1)
    # x축 로그 스케일 추가 (시각화 개선)
    ax.set_xscale('log') 
    
    output_dir = BASE_DIR / "results" / "coherence" / "glitch_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_ch = channel_name.replace(":", "_")
    save_path = output_dir / f"Glitch_{safe_ch}.png"
    plot.savefig(save_path)
    print(f"✅ Saved plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ch", "--channel", required=True)
    args = parser.parse_args()
    
    calc_glitch_coherence(args.channel)
