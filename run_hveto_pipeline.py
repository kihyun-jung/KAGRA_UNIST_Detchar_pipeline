#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAGRA Hveto Pipeline Orchestrator
=================================

Omicron Trigger 결과를 바탕으로 Hveto (Veto Analysis) 파이프라인을 실행합니다.

[Execution Flow]
1. Read Omicron Triggers (from results/omicron/)
2. Generate FFL & Config (with Mock Channels)
3. Submit Hveto Job (Simulation)

Usage:
    python run_hveto_pipeline.py -s 2023-06-18 -e 2023-06-18
"""

import sys
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_ANALYSIS = BASE_DIR / "src" / "analysis"

def run_pipeline(start_date, end_date):
    current_date = start_date
    
    print(f"🚀 Starting Hveto Pipeline ({start_date.date()} ~ {end_date.date()})")
    print("=" * 60)

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        date_str = current_date.strftime("%Y-%m-%d")
        
        print(f"[*] Processing Date: {date_str}")
        
        # 1. Setup Step (setup_hveto.py 실행)
        cmd = [
            "python", str(SRC_ANALYSIS / "setup_hveto.py"),
            "-y", str(year), "-m", str(month), "-d", str(day)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Setup Success: {date_str}")
            
            # 2. Submission Step (Simulation)
            # 실제 condor_submit 대신 메시지 출력
            sub_file = BASE_DIR / "results" / "hveto" / date_str / "submit_hveto.sub"
            print(f"[*] Job Ready: {sub_file}")
            print(f"[*] (Simulation) 'condor_submit {sub_file}' would be executed here.")
            
        except subprocess.CalledProcessError:
            print(f"❌ Setup Failed: {date_str}")
        
        print("-" * 60)
        current_date += timedelta(days=1)

    print("\n🎉 Hveto Pipeline Execution Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 기본값은 우리가 가진 데이터 날짜 (2023-06-18)
    parser.add_argument("-s", "--start", type=str, default="2023-06-18", help="YYYY-MM-DD")
    parser.add_argument("-e", "--end", type=str, default="2023-06-18", help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    s_date = datetime.strptime(args.start, "%Y-%m-%d")
    e_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    run_pipeline(s_date, e_date)
