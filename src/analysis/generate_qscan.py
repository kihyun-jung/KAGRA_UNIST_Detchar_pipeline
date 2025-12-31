#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q-transform Spectrogram Generator
=================================

Hveto 등에서 탐지된 Glitch 구간에 대해
시간-주파수 분석(Q-transform)을 수행하고 이미지를 저장합니다.

[Note]
GitHub Demo 환경에서는 실제 GWF 파일 대신 Random Noise를 생성하여
시각화 파이프라인이 동작함을 보여줍니다.
"""

import os
import matplotlib
matplotlib.use('Agg') # 화면 출력 없이 파일 저장 모드
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

def generate_qscan(channel, gps_time, duration, output_path, snr=0):
    """
    주어진 시간과 채널에 대해 Q-transform 이미지를 생성합니다.
    (Demo: Random Noise 사용)
    """
    # 1. Mock Data 생성 (실제 데이터 로드 대체)
    # 분석 구간: [gps_time - 1, gps_time + 1] (넉넉하게 2초)
    sample_rate = 4096
    epoch = gps_time - 1
    length = int(2 * sample_rate)
    
    # 랜덤 노이즈 생성 (White Noise)
    data = TimeSeries(np.random.randn(length), t0=epoch, sample_rate=sample_rate, name=channel)
    
    # 2. Q-transform 수행 (GWpy 기능 활용)
    # 실제로는 Whiten, Bandpass 등이 필요하지만 데모에서는 기본 q_transform 사용
    try:
        # qrange, frange 등 파라미터 설정
        qspec = data.q_transform(qrange=(4, 64), frange=(10, 1000), 
                                 outseg=(gps_time, gps_time + duration))
        
        # 3. 플로팅
        plot = qspec.plot(figsize=[10, 6])
        ax = plot.gca()
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"Q-scan: {channel} @ {gps_time}")
        ax.colorbar(label='Normalized Energy')

        # SNR 정보 표기 (옵션)
        if snr > 0:
            ax.text(0.95, 0.95, f"SNR: {snr:.2f}", transform=ax.transAxes, 
                    ha='right', va='top', color='white', fontweight='bold')

        # 4. 저장
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot.savefig(output_path)
        plt.close(plot) # 메모리 해제
        
        return True

    except Exception as e:
        print(f"[!] Q-scan generation failed for {channel}: {e}")
        return False

# 테스트용 실행 코드
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", default="K1:MOCK-CHANNEL")
    parser.add_argument("-t", "--time", type=float, default=1371097740.0)
    parser.add_argument("-o", "--output", default="test_qscan.png")
    args = parser.parse_args()
    
    generate_qscan(args.channel, args.time, 0.5, args.output)
    print(f"Test image saved to {args.output}")
