#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Pipeline Orchestrator
======================================
Updated: Switch TensorFlow model format to .keras (Keras 3 compatible)
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ================= Configuration =================
#  scripts/ 폴더 기준 상위 폴더(Project Root)를 찾도록 변경
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC_DIR = PROJECT_ROOT / "src" / "ml"
DATA_DIR = PROJECT_ROOT / "data" / "training_set"
RESULTS_DIR = PROJECT_ROOT / "results"
# =================================================

def run_step(desc, cmd):
    print("=" * 60)
    print(f"[*] Step: {desc}")
    print(f"    Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f" {desc} Completed.")
    except subprocess.CalledProcessError:
        print(f"[!] Failed: {desc}")
        sys.exit(1)

def main(args):
    framework = args.framework
    date_str = f"{args.year}-{args.month:02d}-{args.day:02d}"
    mock_base = f"{date_str}_mock"
    
    # 1. 경로 정의
    target_qscan_dir = RESULTS_DIR / mock_base / "qscans" / "main"
    
    # 결과 폴더
    ml_root_dir = RESULTS_DIR / mock_base / "machine_learning"
    framework_output_dir = ml_root_dir / framework
    
    # Keras 3.0 이상에서는 .keras 포맷이 필수입니다.
    model_extension = ".pth" if framework == "pytorch" else ".keras"
    model_save_path = framework_output_dir / f"model_{framework}{model_extension}"

    # 리포트 파일명
    plot_filename = f"Loss_and_Acc_graph_{framework}.png"
    plot_save_path = framework_output_dir / plot_filename
    
    csv_filename = f"predictions_{framework}.csv"
    csv_save_path = framework_output_dir / csv_filename

    print(f"Starting ML Pipeline using [{framework.upper()}]")
    print(f"    Target Date : {date_str}")
    print(f"    Output Dir  : {framework_output_dir}")
    print(f"    Report File : {csv_filename}")

    # 2. 데이터 유효성 검사
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print(f"[!] Training set not found at {DATA_DIR}")
        print(f"    Please ensure 'data/training_set' contains class folders with images.")
        sys.exit(1)
        
    if not target_qscan_dir.exists():
        print(f"[!] Target Q-scan directory not found: {target_qscan_dir}")
        print(f"    Please run '04_generate_qscan.py' first.")
        sys.exit(1)

    # 폴더 생성
    framework_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 프레임워크별 스크립트 실행
    if framework == "pytorch":
        train_cmd = [
            "python", str(SRC_DIR / "train_pytorch.py"),
            "--data_dir", str(DATA_DIR),
            "--save_path", str(model_save_path),
            "--plot_path", str(plot_save_path),
            "--epochs", "10",
            "--batch_size", "16"
        ]
        
        infer_cmd = [
            "python", str(SRC_DIR / "inference_pytorch.py"),
            "--model_path", str(model_save_path),
            "--input_dir", str(target_qscan_dir),
            "--output_dir", str(framework_output_dir),
            "--csv_path", str(csv_save_path)
        ]
        
    elif framework == "tensorflow":
        train_cmd = [
            "python", str(SRC_DIR / "train_tf.py"),
            "--data_dir", str(DATA_DIR),
            "--save_path", str(model_save_path),
            "--plot_path", str(plot_save_path),
            "--epochs", "10"
        ]
        
        infer_cmd = [
            "python", str(SRC_DIR / "inference_tf.py"),
            "--model_path", str(model_save_path),
            "--input_dir", str(target_qscan_dir),
            "--output_dir", str(framework_output_dir),
            "--csv_path", str(csv_save_path)
        ]

    # 실행
    run_step(f"Training ({framework})", train_cmd)
    run_step(f"Inference ({framework})", infer_cmd)
    
    print("=" * 60)
    print(f" Pipeline Finished. Check results in: {framework_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    parser.add_argument("--framework", type=str, default="pytorch", 
                        choices=["pytorch", "tensorflow"],
                        help="Choose DL framework")
    args = parser.parse_args()
    
    main(args)
