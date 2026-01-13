#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
import re
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.table import EventTable
    from gwpy.segments import Segment
    import numpy as np
except ImportError:
    print("[!] Error: 'gwpy' package is required.")
    sys.exit(1)

# ================= Configuration =================
# RESULTS_DIR 경로를 상위 폴더 기준으로 변경
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MAIN_CHANNEL = "K1:CAL-MOCK" 

# Q-transform 설정
PLOT_DURATION = 0.5   
# =================================================

def get_gwf_file_list(raw_dir):
    files = sorted(list(raw_dir.glob("*.gwf")))
    return [str(f) for f in files]

def make_qscan(task):
    gps, channel, duration, out_dir, gwf_files, label = task
    
    safe_ch_name = channel.replace(':', '_')
    filename = f"{label}-{safe_ch_name}-{gps:.2f}.png"
    out_path = out_dir / filename
    
    if out_path.exists():
        return f"Skipped (Exists): {filename}"

    pad = 16.0 
    start = gps - duration - pad
    end = gps + duration + pad
    
    try:
        # 1. 데이터 읽기
        data = TimeSeries.read(gwf_files, channel, start=start, end=end, format='gwf', nproc=1)
        
        # 2. 주파수 범위 설정
        native_nyquist = data.sample_rate.value / 2.0
        calc_fmax = min(4096, native_nyquist)
        calc_fmin = 8.0 
        
        # 3. Q-transform 계산
        qspec = data.q_transform(
            qrange=(4, 64), 
            frange=(calc_fmin, calc_fmax), 
            gps=gps,
            search=duration, 
            tres=0.002,
            fres=0.5,
            whiten=True,
            outseg=Segment(gps - duration, gps + duration),
            fduration=1,
            highpass=None
        )
        
        # 4. 플롯 그리기
        plot = qspec.plot(figsize=[10, 6], vmin=0, vmax=100)
        ax = plot.gca()
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.set_epoch(gps)
        
        # Y축 상한선 결정
        try:
            maxy = qspec.yindex[-1].value 
        except:
            maxy = float(str(qspec.yindex[-1]).split(' ')[0])
        ax.set_ylim(10, maxy)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{label} | {channel} | GPS: {gps:.2f}")
        ax.colorbar(label='Normalized Energy (Fixed Scale: 0-100)')
        
        plot.savefig(out_path, dpi=100)
        plot.close()
        
        return f"Generated: {filename}"
        
    except Exception as e:
        return f"Failed ({label}-{channel}-{gps}): {e}"

def main():
    parser = argparse.ArgumentParser(description="Generate Q-scans for Hveto Vetoed Triggers")
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    date_str = f"{args.year}-{args.month:02d}-{args.day:02d}"
    mock_base = f"{date_str}_mock"
    
    # RESULTS_DIR 기반 경로 설정
    hveto_dir = RESULTS_DIR / mock_base / "hveto"
    triggers_dir = hveto_dir / "triggers"
    raw_dir = RESULTS_DIR / mock_base / "raw"
    
    # 결과 저장 경로 분리 (Main / Aux)
    qscan_root = RESULTS_DIR / mock_base / "qscans"
    qscan_main_dir = qscan_root / "main"
    qscan_aux_dir = qscan_root / "aux"
    
    print(f"[*] Starting Q-scan Generation for {date_str}")
    
    if not triggers_dir.exists():
        print(f"[!] Hveto triggers directory not found: {triggers_dir}")
        sys.exit(1)
        
    # 폴더 각각 생성
    qscan_main_dir.mkdir(parents=True, exist_ok=True)
    qscan_aux_dir.mkdir(parents=True, exist_ok=True)
    
    gwf_files = get_gwf_file_list(raw_dir)
    if not gwf_files:
        print("[!] No raw GWF files found.")
        sys.exit(1)

    tasks = []
    
    vetoed_files = sorted(list(triggers_dir.glob("K1-HVETO_VETOED_TRIGS_ROUND_*.txt")))
    print(f"[*] Found {len(vetoed_files)} round vetoed files.")
    
    for v_file in vetoed_files:
        try:
            round_str = v_file.name.split('_ROUND_')[1].split('-')[0]
            round_num = int(round_str)
        except: continue
            
        winner_files = list(triggers_dir.glob(f"K1-HVETO_WINNER_TRIGS_ROUND_{round_num}-*.txt"))
        if not winner_files: continue
        w_file = winner_files[0]
        
        print(f"    -> Processing Round {round_num}...")

        try:
            vetoed_tab = EventTable.read(v_file, format='ascii')
            if 'time' not in vetoed_tab.colnames:
                vetoed_tab = EventTable.read(v_file, format='ascii', names=['time', 'frequency', 'q', 'snr', 'tstart', 'tend', 'fstart', 'fend', 'amplitude', 'phase', 'tstart_us', 'channel'])
        except: continue
            
        if len(vetoed_tab) == 0: continue

        try:
            winner_tab = EventTable.read(w_file, format='ascii')
            if len(winner_tab) > 0 and 'channel' in winner_tab.colnames:
                winner_channel = str(winner_tab[0]['channel'])
            else: continue
        except: continue

        print(f"       [+] Winner Channel: {winner_channel}, Vetoed Count: {len(vetoed_tab)}")

        for row in vetoed_tab:
            gps = row['time']
            
            # Task 1: Main Channel
            tasks.append((gps, MAIN_CHANNEL, PLOT_DURATION, qscan_main_dir, gwf_files, f"R{round_num}_Main_Vetoed"))
            
            # Task 2: Winner Channel
            tasks.append((gps, winner_channel, PLOT_DURATION, qscan_aux_dir, gwf_files, f"R{round_num}_Aux_Winner"))

    print(f"[*] Total Q-scans to generate: {len(tasks)}")

    if tasks:
        n_proc = 4 
        with Pool(processes=n_proc) as pool:
            for i, res in enumerate(pool.imap_unordered(make_qscan, tasks), 1):
                sys.stdout.write(f"\r    -> Progress: {i}/{len(tasks)} - {res.split(':')[0]}")
                sys.stdout.flush()
        print("\n [*] All tasks completed.")
        print(f"[*] Results saved in:")
        print(f"    - Main: {qscan_main_dir}")
        print(f"    - Aux : {qscan_aux_dir}")
    else:
        print("[*] No triggers found to plot.")

if __name__ == "__main__":
    main()