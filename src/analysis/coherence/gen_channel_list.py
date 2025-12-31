#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coherence Channel List Generator
================================

파싱된 Hveto 결과 파일들을 읽어,
Coherence 분석 대상이 되는 유니크한 채널 리스트 파일을 생성합니다.
"""

import glob
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

def create_channel_list(output_filename="channel_list.txt"):
    # 입력 디렉토리
    input_dir = BASE_DIR / "results" / "coherence" / "extracted_channels"
    # 출력 파일 경로
    output_path = BASE_DIR / "results" / "coherence" / output_filename
    
    print(f"[*] Generating channel list from {input_dir}...")
    
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print("[!] No extracted channel files found. Run parsing first.")
        return False

    try:
        with open(output_path, 'w') as f:
            for txt_file in txt_files:
                # 파일명이 곧 채널명임 (예: K1_PEM_SENSOR_1.txt)
                channel_name = txt_file.stem
                f.write(f"{channel_name}\n")
                
        print(f"✅ Channel list created: {output_path}")
        print(f"   Total channels: {len(txt_files)}")
        return True

    except Exception as e:
        print(f"[Error] {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="channel_list.txt")
    args = parser.parse_args()
    
    create_channel_list(args.output)
