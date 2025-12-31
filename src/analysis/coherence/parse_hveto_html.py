#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hveto Result Parser
===================

[Input]  results/hveto/{DATE}/index.html
[Output] results/coherence/extracted_channels/{CHANNEL}.txt
"""

import os
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

def generate_date_range(start_str, end_str):
    s_date = datetime.strptime(start_str, "%Y-%m-%d")
    e_date = datetime.strptime(end_str, "%Y-%m-%d")
    date_list = []
    curr = s_date
    while curr <= e_date:
        date_list.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return date_list

def parse_hveto_results(start_date, end_date):
    # 1. 입력 경로 베이스 (Hveto 결과 폴더)
    hveto_base_dir = BASE_DIR / "results" / "hveto"
    
    # 2. 출력 경로 (Coherence 분석용 추출 폴더)
    output_dir = BASE_DIR / "results" / "coherence" / "extracted_channels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_list = generate_date_range(start_date, end_date)
    print(f"[*] Parsing Hveto HTMLs from {start_date} to {end_date}...")
    print(f"[*] Reading from: {hveto_base_dir}/{{DATE}}/index.html")
    
    extracted_count = 0

    for date_str in date_list:
        # 정확한 경로 지정: results/hveto/2023-06-18/index.html
        html_path = hveto_base_dir / date_str / "index.html"
        
        if not html_path.exists():
            print(f"[!] HTML not found: {html_path}")
            continue
            
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            table = soup.find('table', class_='table')
            if not table or not table.find('tbody'):
                print(f"[!] Invalid table structure in {html_path}")
                continue

            for row in table.find('tbody').find_all('tr'):
                cols = row.find_all(['td', 'th'])
                if len(cols) > 4:
                    round_num = cols[0].get_text().strip()
                    channel_raw = cols[1].get_text().strip()
                    sig_raw = cols[4].get_text().strip().split('\n')[0]
                    
                    # 채널명 정제
                    safe_name = re.sub(r'[^\w\s:-]', '', channel_raw).replace(' ', '_')
                    
                    # 월-일 추출
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    mm_dd = dt.strftime("%m-%d")
                    
                    content = f"{mm_dd} {round_num} {sig_raw}"
                    
                    # 파일 저장
                    out_file = output_dir / f"{safe_name}.txt"
                    with open(out_file, 'a', encoding='utf-8') as f:
                        f.write(content + "\n")
                    
                    extracted_count += 1
                    
        except Exception as e:
            print(f"[Error] Failed parsing {date_str}: {e}")

    print(f"[*] Extraction complete. {extracted_count} entries saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    args = parser.parse_args()
    
    parse_hveto_results(args.start, args.end)
