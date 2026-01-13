#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hveto Pipeline Runner with Comprehensive Runtime Hotfixes
=======================================================
Author: Kihyun Jung, Contact: wjk9364@gmail.com
Description: 
    Executes the Hveto analysis pipeline with multiple runtime 'Monkey Patches' 
    to resolve compatibility issues in legacy scientific libraries (hveto, gwpy, matplotlib).

    Applied Fixes:
    1. [Hveto] Core Veto Patch: Fixes IndexError when veto segment list is empty.
    2. [GWPy] ROOT I/O Patch: Forces compatibility between GWPy and Uproot 5.x.
    3. [Matplotlib] Scatter Patch: Resolves 'Normalize instance' conflict error.
    4. [Matplotlib] Axis Limit Patch: Prevents crash on NaN/Inf axis limits.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import traceback
import numpy
import hveto.core
import gwpy.plot.axes
import matplotlib.axes

# ==========================================
# Path Configuration
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# ==========================================
# Monkey Patch 1: Fix IndexError in hveto.core.veto
# ==========================================
def patched_veto(table, segmentlist):
    table.sort('time')
    times = table['time']
    segmentlist = type(segmentlist)(segmentlist).coalesce()
    
    if not segmentlist:
        return table, table[:0]

    keep = numpy.ones(times.shape[0], dtype=bool)
    j = 0
    
    try:
        a, b = segmentlist[j]
    except IndexError:
        return table, table[:0]

    i = 0
    while i < times.size:
        t = times[i]
        if t < a:
            i += 1
            continue
        if t > b:
            j += 1
            try:
                a, b = segmentlist[j]
                continue
            except IndexError:
                break
        keep[i] = False
        i += 1
    return table[keep], table[~keep]

print("[*] Applying Patch 1: hveto.core.veto (Empty Segment Fix)...")
hveto.core.veto = patched_veto

# ==========================================
# Monkey Patch 2: GWPy + Uproot 5.x Fix
# ==========================================
try:
    import uproot
    from gwpy.table import Table, EventTable
    from astropy.io import registry as io_registry

    def patched_table_from_root(source, *args, **kwargs):
        treename = kwargs.get("treename", "triggers")
        if isinstance(source, (str, bytes, os.PathLike)):
            f = uproot.open(source)
        else:
            f = source
        
        if treename not in f:
            for cand in ["triggers", "events", "sngl_burst", "triggers;1"]:
                if cand in f:
                    treename = cand
                    break
        
        if treename not in f:
             raise ValueError(f"Tree '{treename}' not found in {source}. Keys: {f.keys()}")
            
        tree = f[treename]
        return Table(tree.arrays(library="np"))

    io_registry.register_reader('root', Table, patched_table_from_root, force=True)
    io_registry.register_reader('root', EventTable, patched_table_from_root, force=True)
    
    print("[*] Applying Patch 2: GWPy ROOT I/O Registry...")

except ImportError:
    pass

# ==========================================
# Monkey Patch 3: Matplotlib Scatter Fix
# ==========================================
original_scatter = gwpy.plot.axes.Axes.scatter

def safe_scatter(self, x, y, **kwargs):
    if 'norm' in kwargs:
        norm = kwargs['norm']
        if 'vmin' in kwargs:
            val = kwargs.pop('vmin')
            if norm.vmin is None: norm.vmin = val
        if 'vmax' in kwargs:
            val = kwargs.pop('vmax')
            if norm.vmax is None: norm.vmax = val
                
    return original_scatter(self, x, y, **kwargs)

print("[*] Applying Patch 3: Matplotlib Scatter (Norm/Vmin Conflict Fix)...")
gwpy.plot.axes.Axes.scatter = safe_scatter

# ==========================================
# Monkey Patch 4: Matplotlib Axis Limit Fix
# ==========================================
_orig_set_ylim = matplotlib.axes.Axes.set_ylim
_orig_set_xlim = matplotlib.axes.Axes.set_xlim

def safe_set_ylim(self, *args, **kwargs):
    try:
        return _orig_set_ylim(self, *args, **kwargs)
    except ValueError as e:
        if "NaN or Inf" in str(e):
            print(f"    [Patch 4] Caught invalid Y-limits. Resetting to safe defaults (0.1, 100).")
            return _orig_set_ylim(self, 0.1, 100)
        raise e

def safe_set_xlim(self, *args, **kwargs):
    try:
        return _orig_set_xlim(self, *args, **kwargs)
    except ValueError as e:
        if "NaN or Inf" in str(e):
            print(f"    [Patch 4] Caught invalid X-limits. Resetting to safe defaults (0, 1).")
            return _orig_set_xlim(self, 0, 1)
        raise e

print("[*] Applying Patch 4: Matplotlib Axis Limits (NaN/Inf Guard)...")
matplotlib.axes.Axes.set_ylim = safe_set_ylim
matplotlib.axes.Axes.set_xlim = safe_set_xlim

# ==========================================
# Main Logic
# ==========================================
try:
    from gwpy.segments import DataQualityFlag, Segment, SegmentList
except ImportError:
    sys.exit(1)

def check_environment():
    try:
        import hveto
    except ImportError:
        sys.exit(1)

def read_gps_range_from_segments(seg_file):
    try:
        with open(seg_file, "r") as f:
            content = f.read().strip()
            if not content: return None, None
            parts = content.split()
            return float(parts[0]), float(parts[1])
    except FileNotFoundError:
        return None, None

def generate_mock_segment_xml(hveto_out_dir, start, end):
    xml_path = hveto_out_dir / "segments.xml"
    seg = Segment(start, end)
    active_seg_list = SegmentList([seg])
    known_seg_list = SegmentList([seg])
    
    flag = DataQualityFlag(
        name='K1:MOCK-ANALYSIS:1',
        active=active_seg_list,
        known=known_seg_list,
        label='Mock Data Analysis Ready'
    )
    try:
        flag.write(xml_path, format='ligolw')
        return xml_path
    except Exception as e:
        print(f"    [!] Failed to generate segment XML: {e}")
        return None

def generate_hveto_ffl_and_get_channels(omicron_dir, hveto_out_dir, date_str):
    extensions = ["*.root", "*.xml", "*.xml.gz"]
    all_triggers = []
    detected_format = "ligolw" 

    for ext in extensions:
        found = list(omicron_dir.rglob(ext))
        if found:
            all_triggers.extend(found)
            if ".root" in ext:
                detected_format = "root"

    if not all_triggers:
        return None, None, [], None

    primary_list = [r for r in all_triggers if "CAL-MOCK" in r.name or "CAL-MOCK" in str(r.parent)]
    aux_list = [r for r in all_triggers if "AUX-CHANNEL" in r.name or "AUX-CHANNEL" in str(r.parent)]

    if not primary_list:
        return None, None, [], None

    found_aux_channels = set()
    for f in aux_list:
        parent_name = f.parent.name
        if "AUX-CHANNEL" in parent_name:
            found_aux_channels.add(parent_name)

    def write_ffl(path, file_list):
        file_list.sort() 
        with open(path, "w") as f:
            for r in file_list:
                stem = r.name
                for ext in ['.xml.gz', '.xml', '.root']:
                    if stem.endswith(ext):
                        stem = stem[:-len(ext)]
                        break
                parts = stem.split('-')
                try:
                    t_start = parts[-2]
                    dur = parts[-1]
                    f.write(f"{r.resolve()} {t_start} {dur} 0 0\n")
                except (IndexError, ValueError):
                    continue
        return path

    pri_ffl = hveto_out_dir / f"primary_{date_str}.ffl"
    aux_ffl = hveto_out_dir / f"auxiliary_{date_str}.ffl"

    write_ffl(pri_ffl, primary_list)
    write_ffl(aux_ffl, aux_list)
    
    return pri_ffl, aux_ffl, sorted(list(found_aux_channels)), detected_format

def generate_detailed_ini(hveto_out_dir, aux_channels, data_format):
    ini_path = hveto_out_dir / "hveto.ini"
    aux_channel_str = "\n    ".join(aux_channels)

    if data_format == "root":
        read_fmt = "root"
        extra_option = "read-treename = triggers"
    else:
        read_fmt = "ligolw"
        extra_option = "read-tablename = sngl_burst"

    content = f"""[DEFAULT]
ifo = K1

[hveto]
; [Engineering Note]
; SNR Threshold Cap: 50.00
; Reason: Hveto statistically prefers higher thresholds (e.g., 100+).
; However, given the Mock Data distribution, a threshold of 100 resulted in 0 events vetoed (Efficiency 0%),
; leading to infinite loops. Capping at 50 ensures effective convergence.
snr-thresholds = 8.00, 10.00, 15.00, 20.00, 30.00, 50.00
snr-thresholds = 8.00, 10.00, 15.00, 20.00, 30.00, 50.00
time-windows = 0.10, 0.20, 0.40, 0.80, 1.00
minimum-significance = 2.0

[segments]
analysis-flag = K1:MOCK-ANALYSIS:1
padding = 0, -10

[primary]
channel = K1:CAL-MOCK
trigger-generator = Omicron
snr-threshold = 8.0
frequency-range = 10,1000
read-format = {read_fmt}
{extra_option}
read-columns = peak_time, central_freq, snr
cluster-window = 0.5
cluster-rank = snr
cluster-index = time

[auxiliary]
trigger-generator = Omicron
frequency-range = 10,1000
read-format = {read_fmt}
{extra_option}
read-columns = peak_time, central_freq, snr
channels =
    {aux_channel_str}
cluster-window = 0.5
cluster-rank = snr
cluster-index = time

[safety]
unsafe-channels = K1:CAL-MOCK
"""
    with open(ini_path, "w") as f:
        f.write(content)
    return ini_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    args = parser.parse_args()

    check_environment()
    
    date_str = f"{args.year}-{args.month:02d}-{args.day:02d}"
    mock_base = f"{date_str}_mock"
    
    # [수정] RESULTS_DIR 기반 경로 설정
    omicron_dir = RESULTS_DIR / mock_base / "omicron"
    hveto_out_dir = RESULTS_DIR / mock_base / "hveto"
    seg_file = omicron_dir / f"segments_{date_str}_mock.txt"

    print(f"[*] Starting Hveto Analysis for: {date_str}")

    gps_start, gps_end = read_gps_range_from_segments(seg_file)
    if not gps_start:
        sys.exit(1)

    hveto_out_dir.mkdir(parents=True, exist_ok=True)

    seg_xml_path = generate_mock_segment_xml(hveto_out_dir, gps_start, gps_end)
    if not seg_xml_path: sys.exit(1)

    pri_ffl, aux_ffl, detected_channels, data_fmt = generate_hveto_ffl_and_get_channels(omicron_dir, hveto_out_dir, date_str)
    if not pri_ffl: sys.exit(1)
        
    ini_path = generate_detailed_ini(hveto_out_dir, detected_channels, data_fmt)

    hveto_args = [
        "hveto",
        str(int(gps_start)), str(int(gps_end)),
        "--ifo", "K1",
        "--config-file", str(ini_path),
        "--primary-cache", str(pri_ffl),
        "--auxiliary-cache", str(aux_ffl),
        "--analysis-segments", str(seg_xml_path), 
        "--output-directory", str(hveto_out_dir)
    ]
    
    original_argv = sys.argv
    sys.argv = hveto_args
    
    print("    Executing Hveto Engine (Patched Mode)...")
    try:
        from hveto.__main__ import main as hveto_main
        hveto_main()
    except SystemExit as e:
        if e.code != 0:
            print(f"    SystemExit Error Code: {e.code}")
    except Exception as e:
        print(f"    Execution Failed: {e}")
        traceback.print_exc()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
