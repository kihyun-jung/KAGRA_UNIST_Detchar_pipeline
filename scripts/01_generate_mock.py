#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import datetime
import random
import argparse
import numpy as np
from pathlib import Path

try:
    from gwpy.timeseries import TimeSeries, TimeSeriesDict
    from astropy.time import Time
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
except ImportError:
    print("[!] Error: GWpy or Astropy not installed.")
    sys.exit(1)

# ==========================================
# 0. Path Configuration
# ==========================================
# 스크립트 위치: scripts/01_generate_mock.py
# 프로젝트 루트: ../ (scripts의 상위 폴더)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# ==========================================
# 1. Configuration & Channel Setup
# ==========================================
IFO = "K1"
RAW_CHUNK_LEN = 32  
MAIN_CHANNEL = "K1:CAL-MOCK"

FREQ_GROUPS = [512, 1024, 2048, 4096, 8192, 16384]

def setup_channels():
    channel_rates = {MAIN_CHANNEL: 16384}
    aux_channels = []
    for freq in FREQ_GROUPS:
        for i in range(1, 3):
            ch_name = f"K1:AUX-CHANNEL_{freq}_{i}_DQ"
            aux_channels.append(ch_name)
            channel_rates[ch_name] = freq
    return aux_channels, channel_rates

AUX_CHANNELS, CHANNEL_RATES = setup_channels()
ALL_CHANNELS = [MAIN_CHANNEL] + AUX_CHANNELS

# ==========================================
# 2. Logic: Proportional Family Injection
# ==========================================
def create_structured_plan(start_gps, duration_sec):
    plan = []
    end_gps = start_gps + duration_sec
    
    # [설정 1] 시간당 200개
    hourly_rate = 200
    total_events = max(1, int(duration_sec * (hourly_rate / 3600)))
    
    # [설정 2] 패밀리 구성 (보조 채널 랜덤 배정)
    shuffled_aux = random.sample(AUX_CHANNELS, len(AUX_CHANNELS))
    
    # Family 정의 (Leader + Members)
    family_1 = {
        "leader": shuffled_aux[0],
        "members": shuffled_aux[1:3], # 2 members
        "name": "Family_1 (Strong)",
        "ratio": 0.30 
    }
    
    family_2 = {
        "leader": shuffled_aux[3],
        "members": shuffled_aux[4:5], # 1 member
        "name": "Family_2 (Medium)",
        "ratio": 0.10 
    }
    
    family_3 = {
        "leader": shuffled_aux[5],
        "members": shuffled_aux[6:7], # 1 member
        "name": "Family_3 (Weak)",
        "ratio": 0.05 
    }
    
    # 개수 계산
    count_f1 = int(total_events * family_1['ratio'])
    count_f2 = int(total_events * family_2['ratio'])
    count_f3 = int(total_events * family_3['ratio'])
    count_rnd = total_events - (count_f1 + count_f2 + count_f3)
    
    # 패밀리 객체에 개수 주입
    family_1['count'] = count_f1
    family_2['count'] = count_f2
    family_3['count'] = count_f3
    
    print("\n" + "="*60)
    print(f" [PROPORTIONAL INJECTION PLAN]")
    print(f"   Duration: {duration_sec} sec ({duration_sec/3600:.1f} hours)")
    print(f"   Rate: {hourly_rate} events/hour")
    print(f"   Total Events: {total_events}")
    print("-" * 60)
    print(f"    {family_1['name']}: {count_f1} events (50%)")
    print(f"      Leader: {family_1['leader']}")
    print(f"      Members: {family_1['members']}")
    print(f"    {family_2['name']}: {count_f2} events (30%)")
    print(f"      Leader: {family_2['leader']}")
    print(f"    {family_3['name']}: {count_f3} events (10%)")
    print(f"      Leader: {family_3['leader']}")
    print(f"    Random Noise: {count_rnd} events (10%, No Aux)")
    print("="*60 + "\n")

    # 이벤트 생성 함수
    def add_event(fam_type):
        event_time = np.random.uniform(start_gps, end_gps)
        center_freq = random.uniform(60, 300)
        q_value = random.uniform(5, 15)
        
        # 메인 채널SNR
        event = {
            "time": event_time,
            "channels": {
                MAIN_CHANNEL: {"snr": random.uniform(12, 18)}
            },
            "freq": center_freq,
            "q": q_value,
            "type": "structure"
        }
        
        if fam_type:
            # Leader SNR
            event["channels"][fam_type['leader']] = {"snr": random.uniform(15, 25)}
            
            # Members SNR
            for member in fam_type['members']:
                event["channels"][member] = {"snr": random.uniform(8, 12)}
                
        return event

    # 할당량만큼 생성
    for _ in range(count_f1): plan.append(add_event(family_1))
    for _ in range(count_f2): plan.append(add_event(family_2))
    for _ in range(count_f3): plan.append(add_event(family_3))
    for _ in range(count_rnd): plan.append(add_event(None)) # Random
        
    plan.sort(key=lambda x: x['time'])
    return plan

def generate_random_segments(start_gps, duration_sec):
    segments = []
    # 4시간 단위로 세그먼트 나눔 (분석 효율성)
    num_segments = max(1, int(duration_sec / 14400))
    avg_len = duration_sec // num_segments
    current = start_gps
    for _ in range(num_segments):
        end = min(current + int(avg_len * 0.99), start_gps + duration_sec)
        segments.append((current, end))
        current = end + 5
        if current >= start_gps + duration_sec: break
    return segments

def save_segments_to_file(segments, output_dir, date_str):
    file_name = f"segments_{date_str}_mock.txt"
    file_path = output_dir / file_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for start, end in segments:
            f.write(f"{start} {end}\n")
    return file_name

# ==========================================
# 3. GWF Generation
# ==========================================
def generate_raw_gwf(segments, plan, output_dir):
    print(f"[*] Generating Structured GWF files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    
    for seg_start, seg_end in segments:
        current_gps = seg_start
        while current_gps < seg_end:
            chunk_end = min(current_gps + RAW_CHUNK_LEN, seg_end)
            chunk_dur = chunk_end - current_gps
            if chunk_dur <= 0: break
            
            chunk_events = [e for e in plan if current_gps <= e['time'] < chunk_end]
            
            tsd = TimeSeriesDict()
            for channel in ALL_CHANNELS:
                fs = CHANNEL_RATES[channel]
                n_samples = int(chunk_dur * fs)
                
                # 배경 노이즈
                noise_sigma = 1.0
                data = np.random.normal(0, noise_sigma, n_samples)
                
                for event in chunk_events:
                    if channel in event['channels']:
                        ch_info = event['channels'][channel]
                        target_snr = ch_info['snr']
                        target_freq = event['freq']
                        target_q = event['q']
                        
                        if target_freq > (fs / 2.2): continue

                        rel_time = event['time'] - current_gps
                        center_idx = int(rel_time * fs)
                        
                        tau = target_q / (2 * np.pi * target_freq)
                        window_sec = tau * 10
                        width_samples = int(window_sec * fs)
                        t_vec = np.arange(-width_samples, width_samples) / fs
                        
                        # Amp = SNR * Sigma
                        amp = target_snr * noise_sigma
                        
                        # Phase Locking for High Coherence
                        envelope = np.exp(-t_vec**2 / (tau**2))
                        carrier = np.sin(2 * np.pi * target_freq * t_vec)
                        glitch_sig = amp * carrier * envelope
                        
                        start_idx = center_idx - width_samples
                        end_idx = center_idx + width_samples
                        d_start = max(0, start_idx)
                        d_end = min(n_samples, end_idx)
                        s_start = max(0, -start_idx)
                        s_end = s_start + (d_end - d_start)
                        
                        if d_end > d_start:
                            data[d_start:d_end] += glitch_sig[s_start:s_end]
                            
                ts = TimeSeries(data, t0=current_gps, sample_rate=fs, name=channel)
                tsd[channel] = ts
            
            filename = f"{IFO}-RAW_MOCK-{int(current_gps)}-{int(chunk_dur)}.gwf"
            file_path = output_dir / filename
            tsd.write(file_path, format='gwf')
            
            total_files += 1
            current_gps += int(chunk_dur)
            sys.stdout.write(f"\r    -> Generated {total_files} chunks... (GPS: {int(current_gps)})")
            sys.stdout.flush()
            
    print(f"\n    -> Raw Data Generation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    # 하루치 데이터 생성을 원하면 --duration 86400 입력
    parser.add_argument("--duration", type=int, default=14400, help="Duration in sec") 
    args = parser.parse_args()

    try:
        dt = datetime.datetime(args.year, args.month, args.day, tzinfo=datetime.timezone.utc)
        start_gps = int(Time(dt).gps)
        pure_date_str = f"{args.year}-{args.month:02d}-{args.day:02d}"
        
        # RESULTS_DIR 기반 경로 설정
        base_path = RESULTS_DIR / f"{pure_date_str}_mock"
        gwf_dir = base_path / "raw"
        omicron_dir = base_path / "omicron"
        
        print("=" * 60)
        print(f"[*] Target Date: {pure_date_str}")
        print(f"[*] Output Dir : {base_path}")
        
        plan = create_structured_plan(start_gps, args.duration)
        segments = generate_random_segments(start_gps, args.duration)
        
        seg_file_name = save_segments_to_file(segments, omicron_dir, pure_date_str)
        generate_raw_gwf(segments, plan, gwf_dir)
        
        print("\n" + "="*60)
        print(f"[*] Data Ready in: {gwf_dir}")
        print("="*60)
            
    except ValueError as e:
        print(f"[!] Error: {e}")
        sys.exit(1)