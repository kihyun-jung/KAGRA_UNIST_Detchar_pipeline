#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Pipeline Orchestrator (Hybrid)
===============================================

Q-Spectrogram Glitch Classification Pipeline
Supports both PyTorch and TensorFlow frameworks.

Usage:
    python run_ml_pipeline.py --framework pytorch
    python run_ml_pipeline.py --framework tensorflow
"""

import subprocess
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UTILS = BASE_DIR / "src" / "utils"
ML_DIR = BASE_DIR / "src" / "ml"
DATA_DIR = BASE_DIR / "data" / "training_set"

def run_step(desc, cmd):
    print("=" * 60)
    print(f"[*] Step: {desc}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {desc} Completed.")
    except subprocess.CalledProcessError:
        print(f"[!] Failed: {desc}")
        sys.exit(1)

def check_training_data():
    if not DATA_DIR.exists(): return False
    imgs = len(list(DATA_DIR.rglob("*.png")))
    return imgs > 0

def main(framework):
    print(f"🚀 Starting ML Pipeline using [{framework.upper()}]")
    
    # 1. 데이터 확인
    if not check_training_data():
        print("[!] No training data found. Generating MOCK data...")
        run_step("Mock Data Gen", ["python", str(UTILS / "init_ml_data.py")])
    else:
        print("[*] Training data found.")

    # 2. 프레임워크별 스크립트 매핑
    if framework == "pytorch":
        train_script = "train_pytorch.py"
        infer_script = "inference_pytorch.py"
    elif framework == "tensorflow":
        train_script = "train_tf.py"
        infer_script = "inference_tf.py"
    else:
        print("[!] Invalid framework selected.")
        return

    # 3. 학습 및 추론 실행
    run_step(f"Training ({framework})", ["python", str(ML_DIR / train_script)])
    run_step(f"Inference ({framework})", ["python", str(ML_DIR / infer_script)])
    
    print("=" * 60)
    print(f"🎉 Pipeline Finished using {framework}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="pytorch", 
                        choices=["pytorch", "tensorflow"],
                        help="Choose DL framework: 'pytorch' or 'tensorflow'")
    args = parser.parse_args()
    
    main(args.framework)
