#!/usr/bin/env python3
import numpy as np
import sys
import os
import argparse
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.table import EventTable
except ImportError:
    print("[!] Error: 'gwpy' package is required.")
    sys.exit(1)

# ================= Configuration =================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MAIN_CHANNEL_NAME = "K1:CAL-MOCK"

# GWF 파일 단위 설정
GWF_DURATION = 32.0 

# FFT 설정
FFT_LENGTH_SEC = 2.0
OVERLAP_RATIO = 0.5 
# =============================================

def get_gwf_files(raw_dir: Path) -> List[str]:
    if not raw_dir.exists():
        print(f"Raw Directory Not Found: {raw_dir}")
        return []
    files = sorted(list(raw_dir.glob("*.gwf")))
    print(f"Found {len(files)} GWF files in {raw_dir}")
    if files:
        print(f"First GWF: {files[0].name}")
    return [str(f) for f in files]

def get_winner_channel(trigger_dir: Path, round_num: int) -> str:
    pattern = f"K1-HVETO_WINNER_TRIGS_ROUND_{round_num}-*.txt"
    files = list(trigger_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Round {round_num} Winner Trigger file: Not exist.")
    
    target_file = files[0]
    print(f"[*] Found Winner File: {target_file.name}")
    
    try:
        with open(target_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('#'): continue
            if "channel" in line.lower() and "frequency" in line.lower():
                continue
            parts = line.split()
            
            candidate = parts[-1]
            if ":" in candidate:
                return candidate
            
    except Exception as e:
        raise ValueError(f"Winner file parsing failed: {e}")
    
    raise ValueError(f"Winner file ({target_file.name}): No valid information.")

def spectral_density_estimation(main_data, aux_data, fft_duration, overlap_ratio=0.5):
    # window='hann'을 명시하여 PSD와 CSD 간의 계산 조건을 통일
    Pxx = main_data.psd(fftlength=fft_duration, overlap=overlap_ratio, window='hann')
    Pyy = aux_data.psd(fftlength=fft_duration, overlap=overlap_ratio, window='hann')
    Pxy = main_data.csd(aux_data, fftlength=fft_duration, overlap=overlap_ratio, window='hann')
    return Pxx, Pyy, Pxy

def main():
    parser = argparse.ArgumentParser(description="Calculate Overall Coherence (32s Chunking)")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year")
    parser.add_argument("-m", "--month", type=int, required=True, help="Month")
    parser.add_argument("-d", "--day", type=int, required=True, help="Day")
    parser.add_argument("-r", "--round", type=int, default=1, help="Hveto Round Number")
    args = parser.parse_args()

    target_date = f"{args.year}-{args.month:02d}-{args.day:02d}"
    round_num = args.round
    
    print(f"\n --- Overall Coherence Analysis ---")
    print(f"Target Date : {target_date}")
    print(f"Target Round: {round_num}")

    # 1. 경로 설정
    mock_dir = RESULTS_DIR / f"{target_date}_mock"
    hveto_trig_dir = mock_dir / "hveto" / "triggers"
    segment_file = mock_dir / "omicron" / f"segments_{target_date}_mock.txt"
    raw_dir = mock_dir / "raw"
    output_dir = mock_dir / "coherence"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 필수 파일 확인
    if not hveto_trig_dir.exists():
        print(f"[!] Hveto Dir Missing: {hveto_trig_dir}")
        sys.exit(1)
    if not segment_file.exists():
        print(f"[!] Segment File Missing: {segment_file}")
        sys.exit(1)

    # 3. Winner Channel 자동 탐색
    try:
        winner_channel = get_winner_channel(hveto_trig_dir, round_num)
        print(f"Identified Winner Channel: {winner_channel}")
    except Exception as e:
        print(f"[!] Failed to identify winner channel: {e}")
        sys.exit(1)

    # 4. GWF 파일 리스트
    gwf_files = get_gwf_files(raw_dir)
    if not gwf_files:
        print(f"[!] No .gwf files found in {raw_dir}")
        sys.exit(1)

    # 5. Coherence 계산 루프
    Pxx_weighted_sum = None
    Pyy_weighted_sum = None
    Pxy_weighted_sum = None
    
    total_duration = 0.0
    tn_segments = 0
    frequencies = None

    with open(segment_file, 'r') as f:
        seg_lines = f.readlines()

    print(f"[*] Processing segments with {GWF_DURATION}s chunks...")
    
    for sline in seg_lines:
        sline = sline.strip()
        if not sline or sline.startswith('#'): continue

        try:
            seg_start, seg_end = map(float, sline.split())
            
            aligned_start = math.floor(seg_start / GWF_DURATION) * GWF_DURATION
            chunk_start = aligned_start

            while chunk_start < seg_end:
                chunk_end = chunk_start + GWF_DURATION
                
                valid_start = max(chunk_start, seg_start)
                valid_end = min(chunk_end, seg_end)
                duration = valid_end - valid_start
                
                chunk_start = chunk_end

                if duration < FFT_LENGTH_SEC * 2:
                    continue

                try:
                    main_seg = TimeSeries.read(gwf_files, MAIN_CHANNEL_NAME, start=valid_start, end=valid_end, format='gwf', nproc=1)
                    aux_seg = TimeSeries.read(gwf_files, winner_channel, start=valid_start, end=valid_end, format='gwf', nproc=1)

                    if main_seg.sample_rate.value != aux_seg.sample_rate.value:
                        target_rate = min(main_seg.sample_rate.value, aux_seg.sample_rate.value)
                        if main_seg.sample_rate.value > target_rate:
                            main_seg = main_seg.resample(target_rate)
                        if aux_seg.sample_rate.value > target_rate:
                            aux_seg = aux_seg.resample(target_rate)

                    Pxx, Pyy, Pxy = spectral_density_estimation(main_seg, aux_seg, FFT_LENGTH_SEC, OVERLAP_RATIO)

                    if frequencies is None:
                        frequencies = Pxx.frequencies

                    weight = duration
                    if Pxx_weighted_sum is None:
                        Pxx_weighted_sum = Pxx * weight
                        Pyy_weighted_sum = Pyy * weight
                        Pxy_weighted_sum = Pxy * weight
                    else:
                        Pxx_weighted_sum += Pxx * weight
                        Pyy_weighted_sum += Pyy * weight
                        Pxy_weighted_sum += Pxy * weight

                    total_duration += duration
                    tn_segments += 1
                    
                    sys.stdout.write(f"\r  -> Processing Chunk: {valid_start:.0f}-{valid_end:.0f} ({duration:.1f}s)")
                    sys.stdout.flush()

                except Exception as e:
                    pass

        except Exception as e:
            print(f"\n [!] Segment Parsing Error: {e}")
            continue

    print(f"\n\n [*] Processing Finished.")
    print(f"    Total Valid Chunks: {tn_segments}")
    print(f"    Total Duration    : {total_duration:.2f} sec")

    if total_duration == 0 or Pxx_weighted_sum is None:
        print("[!] No valid data processed.")
        sys.exit(1)

    Pxx_avg = Pxx_weighted_sum / total_duration
    Pyy_avg = Pyy_weighted_sum / total_duration
    Pxy_avg = Pxy_weighted_sum / total_duration

    # Coherence 계산
    coh_value = (np.abs(Pxy_avg.value)**2) / (Pxx_avg.value * Pyy_avg.value)
    
    # 1.0을 넘는 수치적 오차 제거 (Clipping)
    coh_value = np.clip(coh_value, 0, 1.0)

    overall_coh = FrequencySeries(
        coh_value,
        frequencies=Pxx_avg.frequencies,
        unit=None
    )

    safe_ch_name = winner_channel.replace(":", "_")
    output_filename = output_dir / f"Overall_Coherence_{target_date}_R{round_num}_{safe_ch_name}.png"
    
    x_min = 10.0
    x_max = frequencies.max().value

    plot = overall_coh.plot(figsize=[12, 6], color='C1')
    ax = plot.gca()
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Coherence')
    
    # Y축은 0~1로 고정
    ax.set_ylim(0, 1.05) 
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"Overall Coherence (Round {round_num} Winner)\nMain: {MAIN_CHANNEL_NAME} vs {winner_channel}", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, which='both')

    plot.savefig(output_filename, dpi=100)
    plot.close()
    
    print(f"Result saved: {output_filename}")

if __name__ == "__main__":
    main()
