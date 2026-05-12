s#!/usr/bin/env python3
"""
汇总所有实验结果，包括 SIREN 对比
Usage: python summarize_all_results.py
"""

import json
import glob
import math

def f(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{n}f}"
    return str(x)

def pct(old, new, lower=False):
    if old is None or new is None or old == 0:
        return None
    if lower:
        return (old - new) / abs(old) * 100
    return (new - old) / abs(old) * 100

def dataset_from_path(path):
    p = path.lower()
    if "ils" in p:
        return "ILS"
    if "mec" in p:
        return "MEC"
    return None

def crop_from_path(path):
    p = path.lower()
    if "crop512" in p:
        return "crop512"
    if "crop256" in p:
        return "crop256"
    return "crop?"

def log_from_path(path):
    p = path.lower()
    if "log16" in p:
        return "log16"
    if "log15" in p:
        return "log15"
    return "log?"

def method_from(path, result):
    m = result.get("method")
    p = path.lower()
    
    if "siren" in p:
        return "COIN (SIREN)"
    
    if m == "Hash-INR baseline":
        return "Hash-INR"
    if m == "HGP-INR baseline":
        return "HGP-INR"
    if m == "COIN (SIREN)":
        return "COIN (SIREN)"
    if m:
        return m
    
    if "poreguard_hgp" in p or ("poreguard" in p and "hgp" in p):
        return "PoreGuard-HGP"
    if "hgp" in p:
        return "HGP-INR"
    if "poreguard" in p or "topology" in p:
        return "PoreGuard-INR"
    if "hash" in p or "baseline" in p:
        return "Hash-INR"
    return "UNKNOWN"

def variant_from(path):
    p = path.lower()
    if "boundary_only" in p:
        return "boundary_only"
    if "throat_only" in p:
        return "throat_only"
    if "smallplane" in p:
        return "smallplane"
    if "bw1p5_tw1" in p:
        return "bw1.5_tw1"
    if "bw3_tw1" in p:
        return "bw3_tw1"
    if "bw3_tw2" in p:
        return "bw3_tw2"
    if "bw3_tw4" in p:
        return "bw3_tw4"
    return "-"

# Load all results
records = []
for path in sorted(glob.glob("output/*/results.json")):
    try:
        with open(path) as fobj:
            r = json.load(fobj)
    except:
        print(f"Warning: Failed to load {path}")
        continue
    
    vals = r.get("val_metrics", [])
    if not vals:
        continue
    
    m = vals[-1]
    dataset = dataset_from_path(path)
    if dataset is None:
        continue
    
    records.append({
        "path": path,
        "dataset": dataset,
        "crop": crop_from_path(path),
        "log": log_from_path(path),
        "method": method_from(path, r),
        "variant": variant_from(path),
        "cr": r.get("compression_ratio"),
        "size": r.get("model_size_mb"),
        "psnr": m.get("psnr"),
        "ssim": m.get("ssim"),
        "acc": m.get("label_accuracy"),
        "por": m.get("porosity_error"),
        "conn": m.get("connectivity_error"),
        "comp": m.get("component_count_error"),
    })

method_order = {
    "Hash-INR": 0,
    "COIN (SIREN)": 1,
    "PoreGuard-INR": 2,
    "HGP-INR": 3,
    "PoreGuard-HGP": 4,
}

records.sort(key=lambda x: (
    x["dataset"], x["crop"], x["log"],
    method_order.get(x["method"], 99), x["variant"], x["path"]
))

# Print summary
print("\n" + "=" * 130)
print("SUMMARY TABLE - ALL METHODS")
print("=" * 130)
print(f"{'Dataset':<8} {'Crop':<8} {'Log':<6} {'Method':<18} {'Variant':<16} "
      f"{'CR':>6} {'Size':>6} {'PSNR':>7} {'SSIM':>7} {'Acc':>7} "
      f"{'PorErr':>9} {'ConnErr':>9} {'CompErr':>8}")
print("-" * 130)

for r in records:
    print(f"{r['dataset']:<8} {r['crop']:<8} {r['log']:<6} {r['method']:<18} {r['variant']:<16} "
          f"{f(r['cr'], 2):>6} {f(r['size'], 2):>6} {f(r['psnr'], 2):>7} "
          f"{f(r['ssim'], 4):>7} {f(r['acc'], 4):>7} "
          f"{f(r['por'], 6):>9} {f(r['conn'], 6):>9} {f(r['comp'], 0):>8}")

def find_one(dataset, crop, log, method, variant=None):
    candidates = [r for r in records if r["dataset"] == dataset and r["crop"] == crop 
                  and r["log"] == log and r["method"] == method]
    if variant is not None:
        candidates = [r for r in candidates if r["variant"] == variant]
    if not candidates:
        return None
    priority = {"bw1.5_tw1": 0, "smallplane": 1, "-": 2, "bw3_tw2": 3, "bw3_tw1": 4, 
                "bw3_tw4": 5, "boundary_only": 6, "throat_only": 7}
    return sorted(candidates, key=lambda r: (priority.get(r["variant"], 99), r["path"]))[0]

def compare(title, base, target):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if base is None or target is None:
        print("  Missing records.")
        return
    print(f"  Base:   {base['method']:<18} ({base['variant']:<16}) | {base['path']}")
    print(f"  Target: {target['method']:<18} ({target['variant']:<16}) | {target['path']}")
    print("-" * 110)
    for key, label, lower in [("psnr", "PSNR", False), ("ssim", "SSIM", False), 
                               ("acc", "Accuracy", False), ("por", "Porosity error", True), 
                               ("conn", "Connectivity error", True), ("comp", "Component count error", True)]:
        old = base.get(key)
        new = target.get(key)
        if old is None or new is None:
            continue
        change = pct(old, new, lower)
        direction = "improved" if change is not None and change > 0 else "worse"
        print(f"  {label:20s}: {f(old, 6):>10} -> {f(new, 6):>10} | Δ={f(change, 2):>7}%  {direction}")

print("\n\n" + "=" * 130)
print("KEY COMPARISONS - INCLUDING SIREN")
print("=" * 130)

ils_hash15 = find_one("ILS", "crop256", "log15", "Hash-INR")
ils_siren15 = find_one("ILS", "crop256", "log15", "COIN (SIREN)")
ils_pg15 = find_one("ILS", "crop256", "log15", "PoreGuard-INR", "bw1.5_tw1")
ils_hgp_small15 = find_one("ILS", "crop256", "log15", "HGP-INR", "smallplane")
ils_pg_hgp_small15 = find_one("ILS", "crop256", "log15", "PoreGuard-HGP", "smallplane")

if ils_siren15:
    compare("ILS crop256 log15: Hash-INR vs COIN (SIREN)", ils_hash15, ils_siren15)
compare("ILS crop256 log15: Hash-INR vs PoreGuard-INR", ils_hash15, ils_pg15)
compare("ILS crop256 log15: Hash-INR vs HGP-INR smallplane", ils_hash15, ils_hgp_small15)
compare("ILS crop256 log15: Hash-INR vs PoreGuard-HGP smallplane", ils_hash15, ils_pg_hgp_small15)

mec_hash15 = find_one("MEC", "crop256", "log15", "Hash-INR")
mec_siren15 = find_one("MEC", "crop256", "log15", "COIN (SIREN)")
mec_pg15 = find_one("MEC", "crop256", "log15", "PoreGuard-INR")
mec_hash16 = find_one("MEC", "crop256", "log16", "Hash-INR")
mec_pg_hgp16 = find_one("MEC", "crop256", "log16", "PoreGuard-HGP", "smallplane")

if mec_siren15:
    compare("MEC crop256 log15: Hash-INR vs COIN (SIREN)", mec_hash15, mec_siren15)
compare("MEC crop256 log15: Hash-INR vs PoreGuard-INR", mec_hash15, mec_pg15)
compare("MEC crop256 log16: Hash-INR vs PoreGuard-HGP smallplane", mec_hash16, mec_pg_hgp16)

print("\n\n" + "=" * 130)
print("PAPER TABLE: ILS CROP256 LOG15 COMPARISON")
print("=" * 130)
print(f"{'Method':<20} {'Variant':<16} {'CR':>6} {'Size':>6} {'PSNR':>7} {'SSIM':>7} "
      f"{'Acc':>7} {'PorErr':>9} {'ConnErr':>9} {'CompErr':>8}")
print("-" * 130)

paper_methods = [
    ("Hash-INR", "-"),
    ("COIN (SIREN)", "-"),
    ("PoreGuard-INR", "bw1.5_tw1"),
    ("HGP-INR", "smallplane"),
    ("PoreGuard-HGP", "smallplane"),
]

for method, variant in paper_methods:
    r = find_one("ILS", "crop256", "log15", method, variant)
    if r:
        print(f"{r['method']:<20} {r['variant']:<16} {f(r['cr'], 2):>6} {f(r['size'], 2):>6} "
              f"{f(r['psnr'], 2):>7} {f(r['ssim'], 4):>7} {f(r['acc'], 4):>7} "
              f"{f(r['por'], 6):>9} {f(r['conn'], 6):>9} {f(r['comp'], 0):>8}")

print("\n" + "=" * 130)
print(f"Total experiments: {len(records)}")
print("=" * 130)
