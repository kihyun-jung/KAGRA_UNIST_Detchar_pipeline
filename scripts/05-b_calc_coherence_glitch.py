#!/usr/bin/env python3
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.frequencyseries import FrequencySeries
except ImportError:
    print("[!] Error: 'gwpy' package is required.")
    sys.exit(1)

# ================= Configuration =================
# RESULTS_DIR 경로를 상위 폴더 기준으로 변경
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MAIN_CHANNEL_NAME = "K1:CAL-MOCK"

# Glitch Coherence 설정
GLITCH_WINDOW_DURATION = 0.5
FFT_LENGTH_SEC = 0.5 
OVERLAP_RATIO = 0.5
# =============================================

def get_gwf_files(raw_dir: Path) -> List[str]:
    """Mock Raw Data (.gwf) 파일 리스트 반환"""
    if not raw_dir.exists():
        return []
    files = sorted(list(raw_dir.glob("*.gwf")))
    return [str(f) for f in files]

def get_winner_channel(trigger_dir: Path, round_num: int) -> str:
    """Winner Channel 자동 탐색"""
    pattern = f"K1-HVETO_WINNER_TRIGS_ROUND_{round_num}-*.txt"
    files = list(trigger_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Round {round_num} : No Winner Trigger file.")
    
    target_file = files[0]
    print(f"[*] Found Winner File: {target_file.name}")
    
    try:
        with open(target_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if "channel" in line.lower() and "frequency" in line.lower(): continue
            
            parts = line.split()
            candidate = parts[-1]
            if ":" in candidate:
                return candidate
            
    except Exception as e:
        raise ValueError(f"Winner file parsing failed: {e}")
    
    raise ValueError(f"Winner file ({target_file.name}): No valid information.")

def get_vetoed_trigger_file(trigger_dir: Path, round_num: int) -> Path:
    """Hveto Vetoed Trigger 파일 찾기"""
    pattern = f"K1-HVETO_VETOED_TRIGS_ROUND_{round_num}-*.txt"
    files = list(trigger_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Round {round_num} Vetoed Trigger file: Not exist")
    
    return files[0]

def parse_triggers(trigger_file: Path) -> List[Tuple[float, float]]:
    """트리거 파일 파싱"""
    triggers = []
    print(f"[*] Parsing vetoed triggers from: {trigger_file.name}")
    
    try:
        with open(trigger_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = line.split()
            try:
                gps_time = float(parts[0]) 
                snr = float(parts[3])       
                triggers.append((gps_time, snr))
            except ValueError:
                continue
                
    except Exception as e:
        print(f"[!] Error parsing trigger file: {e}")
        
    return triggers

# Robust 로직 적용
def spectral_density_estimation(main_data: TimeSeries, aux_data: TimeSeries, fft_duration: float, overlap_ratio: float = 0.5):
    """
    gwpy TimeSeries의 .psd()와 .csd() 메서드를 사용하여 Pxx, Pyy, Pxy를 계산하고 반환합니다.
    데이터 길이가 짧아 'noverlap must be less than nperseg' 오류가 발생할 경우, overlap_ratio를 0.0으로 줄여 재시도합니다.
    """
    if not len(main_data) or not len(aux_data):
        raise ValueError("Data is too short or empty.")

    # 데이터의 샘플링 레이트 확인 및 리샘플링을 통해 통일
    main_rate = main_data.sample_rate.value
    aux_rate = aux_data.sample_rate.value
    target_rate = min(main_rate, aux_rate)

    if main_rate != aux_rate:
        main_to_process = main_data.resample(target_rate)
        aux_to_process = aux_data.resample(target_rate)
    else:
        main_to_process = main_data
        aux_to_process = aux_data

    def calculate_spectra(data1, data2, fftlen, overlap_r):
        """실제 스펙트럼 계산 로직"""
        Pxx = data1.psd(fftlength=fftlen, overlap=overlap_r, window='hann')
        Pyy = data2.psd(fftlength=fftlen, overlap=overlap_r, window='hann')
        Pxy = data1.csd(data2, fftlength=fftlen, overlap=overlap_r, window='hann')
        return Pxx, Pyy, Pxy

    try:
        # 1차 시도: 요청된 오버랩 비율 사용
        Pxx, Pyy, Pxy = calculate_spectra(main_to_process, aux_to_process, fft_duration, overlap_ratio)

    except ValueError as e:
        error_message = str(e)
        # gwpy/scipy 버전마다 에러 메시지가 조금씩 다를 수 있어 핵심 키워드로 확인
        if "noverlap" in error_message or "segment" in error_message:
            # 2차 시도: 오버랩 비율을 0.0으로 줄여 재시도
            try:
                Pxx, Pyy, Pxy = calculate_spectra(main_to_process, aux_to_process, fft_duration, 0.0)
            except ValueError as retry_e:
                raise ValueError(f"GWpy spectral calculation failed even with overlap=0.0: {retry_e}")
        else:
            raise ValueError(f"GWpy spectral calculation failed: {e}")

    return Pxx, Pyy, Pxy

def main():
    parser = argparse.ArgumentParser(description="Calculate SNR-Weighted Glitch Coherence (Robust)")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year")
    parser.add_argument("-m", "--month", type=int, required=True, help="Month")
    parser.add_argument("-d", "--day", type=int, required=True, help="Day")
    parser.add_argument("-r", "--round", type=int, default=1, help="Hveto Round Number")
    args = parser.parse_args()

    target_date = f"{args.year}-{args.month:02d}-{args.day:02d}"
    round_num = args.round
    
    print(f"\n --- Glitch Coherence Analysis (Robust Mode) ---")
    print(f"Target Date : {target_date}")
    print(f"Target Round: {round_num}")

    # 1. 경로 설정
    mock_dir = RESULTS_DIR / f"{target_date}_mock"
    hveto_trig_dir = mock_dir / "hveto" / "triggers"
    raw_dir = mock_dir / "raw"
    output_dir = mock_dir / "coherence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 파일 확인
    if not hveto_trig_dir.exists():
        print(f"[!] Hveto trigger directory missing: {hveto_trig_dir}")
        sys.exit(1)
        
    gwf_files = get_gwf_files(raw_dir)
    if not gwf_files:
        print(f"[!] No GWF files found in {raw_dir}")
        sys.exit(1)

    # 3. 채널 탐색
    try:
        aux_channel = get_winner_channel(hveto_trig_dir, round_num)
        print(f"Identified Winner Channel: {aux_channel}")
    except Exception as e:
        print(f"[!] Failed to identify winner channel: {e}")
        sys.exit(1)

    # 4. 트리거 로드
    try:
        trigger_path = get_vetoed_trigger_file(hveto_trig_dir, round_num)
        triggers = parse_triggers(trigger_path)
    except Exception as e:
        print(f"[!] {e}")
        sys.exit(1)

    if not triggers:
        print("[!] No triggers found.")
        sys.exit(1)

    print(f"[*] Total Glitch Triggers: {len(triggers)}")

    # 5. 계산 루프
    Pxx_w_sum = None
    Pyy_w_sum = None
    Pxy_w_sum = None
    total_snr = 0.0
    processed_count = 0
    frequencies = None
    
    print(f"[*] Calculating Coherence for each glitch (Window: {GLITCH_WINDOW_DURATION}s)...")

    for gps_time, snr in triggers:
        start_time = gps_time - (GLITCH_WINDOW_DURATION / 2.0)
        end_time = gps_time + (GLITCH_WINDOW_DURATION / 2.0)

        try:
            # 데이터 로드
            main_seg = TimeSeries.read(gwf_files, MAIN_CHANNEL_NAME, start=start_time, end=end_time, format='gwf', nproc=1)
            aux_seg = TimeSeries.read(gwf_files, aux_channel, start=start_time, end=end_time, format='gwf', nproc=1)

            # 스펙트럼 계산 (Robust 함수 호출 - 내부에서 리샘플링 및 재시도 수행)
            Pxx, Pyy, Pxy = spectral_density_estimation(main_seg, aux_seg, FFT_LENGTH_SEC, OVERLAP_RATIO)

            if frequencies is None:
                frequencies = Pxx.frequencies

            weight = snr
            
            if Pxx_w_sum is None:
                Pxx_w_sum = Pxx * weight
                Pyy_w_sum = Pyy * weight
                Pxy_w_sum = Pxy * weight
            else:
                Pxx_w_sum += Pxx * weight
                Pyy_w_sum += Pyy * weight
                Pxy_w_sum += Pxy * weight

            total_snr += weight
            processed_count += 1
            
            if processed_count % 10 == 0:
                sys.stdout.write(f"\r  -> Processed {processed_count}/{len(triggers)} triggers")
                sys.stdout.flush()

        except Exception as e:
            # 재시도까지 실패했거나 데이터가 아예 없는 경우에만 스킵
            # print(f"{e}")
            continue

    print(f"\n\n [*] Calculation Finished.")
    print(f"    Processed Triggers: {processed_count}")
    print(f"    Total SNR Weight  : {total_snr:.2f}")

    if total_snr == 0 or Pxx_w_sum is None:
        print("[!] Failed to calculate coherence. No valid triggers processed.")
        sys.exit(1)

    # 6. 최종 결과 계산
    Pxx_avg = Pxx_w_sum / total_snr
    Pyy_avg = Pyy_w_sum / total_snr
    Pxy_avg = Pxy_w_sum / total_snr

    coh_value = (np.abs(Pxy_avg.value)**2) / (Pxx_avg.value * Pyy_avg.value)
    
    glitch_coh = FrequencySeries(
        coh_value,
        frequencies=Pxx_avg.frequencies,
        unit=None
    )

    safe_ch_name = aux_channel.replace(":", "_")
    output_filename = output_dir / f"Glitch_Coherence_{target_date}_R{round_num}_{safe_ch_name}.png"
    
    x_min = 10.0
    x_max = frequencies.max().value

    plot = glitch_coh.plot(figsize=[12, 6], color='C3')
    ax = plot.gca()
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Coherence')
    ax.set_ylim(0, 1)
    ax.set_xlim(x_min, x_max)
    
    title_str = (f"SNR-Weighted Glitch Coherence (Round {round_num} Winner)\n"
                 f"Main: {MAIN_CHANNEL_NAME} vs {aux_channel}\n"
                 f"Events: {processed_count}, Window: {GLITCH_WINDOW_DURATION}s")
    ax.set_title(title_str, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, which='both')

    plot.savefig(output_filename, dpi=100)
    plot.close()
    
    print(f"Result saved: {output_filename}")

if __name__ == "__main__":
    main()