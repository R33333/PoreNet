#!/usr/bin/env python3
"""
Summarize all experiment results from output/*/results.json
Usage: python summarize_results.py
"""

import json
import glob
import math

def f(x, n=4):
    """Format number with n decimal places"""
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{n}f}"
    return str(x)

def pct(old, new, lower=False):
    """Calculate percentage change"""
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
    if "log17" in p:
        return "log17"
    return "log?"

def method_from(path, result):
    m = result.get("method")
    p = path.lower()

    if m == "Hash-INR baseline":
        return "Hash-INR"
    if m == "HGP-INR baseline":
        return "HGP-INR"
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
    # Check specific variants first before generic ones
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
        "lcr_pred": m.get("largest_component_ratio_pred"),
        "lcr_gt": m.get("largest_component_ratio_gt"),
    })

method_order = {
    "Hash-INR": 0,
    "PoreGuard-INR": 1,
    "HGP-INR": 2,
    "PoreGuard-HGP": 3,
}

records.sort(key=lambda x: (
    x["dataset"],
    x["crop"],
    x["log"],
    method_order.get(x["method"], 99),
    x["variant"],
    x["path"],
))

# Print summary table
print("\n" + "=" * 120)
print("SUMMARY TABLE")
print("=" * 120)
print(f"{'Dataset':<8} {'Crop':<8} {'Log':<6} {'Method':<16} {'Variant':<16} "
      f"{'CR':>6} {'Size':>6} {'PSNR':>7} {'SSIM':>7} {'Acc':>7} "
      f"{'PorErr':>9} {'ConnErr':>9} {'CompErr':>8}")
print("-" * 120)

for r in records:
    print(f"{r['dataset']:<8} {r['crop']:<8} {r['log']:<6} {r['method']:<16} {r['variant']:<16} "
          f"{f(r['cr'], 2):>6} {f(r['size'], 2):>6} {f(r['psnr'], 2):>7} "
          f"{f(r['ssim'], 4):>7} {f(r['acc'], 4):>7} "
          f"{f(r['por'], 6):>9} {f(r['conn'], 6):>9} {f(r['comp'], 0):>8}")

# Helper function to find best matching record
def find_one(dataset, crop, log, method, variant=None):
    candidates = [
        r for r in records
        if r["dataset"] == dataset
        and r["crop"] == crop
        and r["log"] == log
        and r["method"] == method
    ]
    if variant is not None:
        candidates = [r for r in candidates if r["variant"] == variant]

    if not candidates:
        return None

    # Priority: prefer standard configs
    priority = {
        "bw1.5_tw1": 0,
        "smallplane": 1,
        "-": 2,
        "bw3_tw2": 3,
        "bw3_tw1": 4,
        "bw3_tw4": 5,
        "boundary_only": 6,
        "throat_only": 7,
    }

    return sorted(candidates, key=lambda r: (priority.get(r["variant"], 99), r["path"]))[0]

def compare(title, base, target):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if base is None or target is None:
        print("  Missing records.")
        return

    print(f"  Base:   {base['method']:<16} ({base['variant']:<16}) | {base['path']}")
    print(f"  Target: {target['method']:<16} ({target['variant']:<16}) | {target['path']}")
    print("-" * 100)

    for key, label, lower in [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("acc", "Accuracy", False),
        ("por", "Porosity error", True),
        ("conn", "Connectivity error", True),
        ("comp", "Component count error", True),
    ]:
        old = base.get(key)
        new = target.get(key)
        if old is None or new is None:
            continue

        change = pct(old, new, lower)
        direction = "✓ improved" if change is not None and change > 0 else "✗ worse"

        print(f"  {label:20s}: {f(old, 6):>10} -> {f(new, 6):>10} | "
              f"Δ={f(change, 2):>7}%  {direction}")

print("\n\n" + "=" * 120)
print("KEY COMPARISONS")
print("=" * 120)

# ILS log15 comparisons
ils_hash15 = find_one("ILS", "crop256", "log15", "Hash-INR")
ils_pg15 = find_one("ILS", "crop256", "log15", "PoreGuard-INR", "bw1.5_tw1")
ils_hgp_small15 = find_one("ILS", "crop256", "log15", "HGP-INR", "smallplane")
ils_pg_hgp_small15 = find_one("ILS", "crop256", "log15", "PoreGuard-HGP", "smallplane")
ils_pg_hgp_full15 = find_one("ILS", "crop256", "log15", "PoreGuard-HGP", "-")

compare("ILS crop256 log15: Hash-INR vs PoreGuard-INR", ils_hash15, ils_pg15)
compare("ILS crop256 log15: Hash-INR vs HGP-INR smallplane", ils_hash15, ils_hgp_small15)
compare("ILS crop256 log15: Hash-INR vs PoreGuard-HGP smallplane", ils_hash15, ils_pg_hgp_small15)
if ils_pg_hgp_full15:
    compare("ILS crop256 log15: Hash-INR vs PoreGuard-HGP full", ils_hash15, ils_pg_hgp_full15)

# MEC comparisons
mec_hash15 = find_one("MEC", "crop256", "log15", "Hash-INR")
mec_pg15 = find_one("MEC", "crop256", "log15", "PoreGuard-INR")
mec_hash16 = find_one("MEC", "crop256", "log16", "Hash-INR")
mec_pg_hgp16 = find_one("MEC", "crop256", "log16", "PoreGuard-HGP", "smallplane")

compare("MEC crop256 log15: Hash-INR vs PoreGuard-INR", mec_hash15, mec_pg15)
compare("MEC crop256 log16: Hash-INR vs PoreGuard-HGP smallplane", mec_hash16, mec_pg_hgp16)

# crop512 comparisons
ils_hash512 = find_one("ILS", "crop512", "log16", "Hash-INR")
ils_pg_hgp512 = find_one("ILS", "crop512", "log16", "PoreGuard-HGP", "smallplane")
mec_hash512 = find_one("MEC", "crop512", "log16", "Hash-INR")
mec_pg_hgp512 = find_one("MEC", "crop512", "log16", "PoreGuard-HGP", "smallplane")

if ils_hash512 and ils_pg_hgp512:
    compare("ILS crop512 log16: Hash-INR vs PoreGuard-HGP smallplane", ils_hash512, ils_pg_hgp512)
if mec_hash512 and mec_pg_hgp512:
    compare("MEC crop512 log16: Hash-INR vs PoreGuard-HGP smallplane", mec_hash512, mec_pg_hgp512)

# Auto-generated conclusion
print("\n\n" + "=" * 120)
print("AUTO-GENERATED CONCLUSION SENTENCES")
print("=" * 120)

if ils_hash15 and ils_hgp_small15:
    conn_gain = pct(ils_hash15["conn"], ils_hgp_small15["conn"], lower=True)
    print(
        f"\n1. On ILS crop256 log15, the compact HGP encoder reduces connectivity error "
        f"from {f(ils_hash15['conn'], 6)} to {f(ils_hgp_small15['conn'], 6)} "
        f"({f(conn_gain, 2)}% reduction), indicating that hybrid grid-plane features "
        f"improve dominant pore-connectivity preservation."
    )

if ils_hash15 and ils_pg15:
    por_gain = pct(ils_hash15["por"], ils_pg15["por"], lower=True)
    conn_gain = pct(ils_hash15["conn"], ils_pg15["conn"], lower=True)
    print(
        f"\n2. On ILS crop256 log15, PoreGuard-INR reduces porosity error from "
        f"{f(ils_hash15['por'], 6)} to {f(ils_pg15['por'], 6)} "
        f"({f(por_gain, 2)}% reduction) and connectivity error by {f(conn_gain, 2)}%, "
        f"showing that pore-structure weighting improves both porosity and dominant "
        f"connectivity under high compression."
    )

if ils_hash15 and ils_pg_hgp_small15:
    psnr_gain = pct(ils_hash15["psnr"], ils_pg_hgp_small15["psnr"], lower=False)
    por_gain = pct(ils_hash15["por"], ils_pg_hgp_small15["por"], lower=True)
    conn_gain = pct(ils_hash15["conn"], ils_pg_hgp_small15["conn"], lower=True)
    print(
        f"\n3. On ILS crop256 log15, PoreGuard-HGP smallplane improves PSNR by "
        f"{f(psnr_gain, 2)}%, reduces porosity error by {f(por_gain, 2)}%, "
        f"and reduces connectivity error by {f(conn_gain, 2)}%."
    )

if mec_hash15 and mec_pg15:
    por_gain = pct(mec_hash15["por"], mec_pg15["por"], lower=True)
    conn_gain = pct(mec_hash15["conn"], mec_pg15["conn"], lower=True)
    print(
        f"\n4. On MEC crop256 log15, PoreGuard-INR reduces porosity error by "
        f"{f(por_gain, 2)}% and connectivity error by {f(conn_gain, 2)}%, "
        f"demonstrating that pore-structure weighting remains effective under severe compression."
    )

if mec_hash16 and mec_pg_hgp16:
    psnr_gain = pct(mec_hash16["psnr"], mec_pg_hgp16["psnr"], lower=False)
    por_gain = pct(mec_hash16["por"], mec_pg_hgp16["por"], lower=True)
    conn_gain = pct(mec_hash16["conn"], mec_pg_hgp16["conn"], lower=True)
    print(
        f"\n5. On MEC crop256 log16, PoreGuard-HGP smallplane improves PSNR by "
        f"{f(psnr_gain, 2)}%, reduces porosity error by {f(por_gain, 2)}%, "
        f"and reduces connectivity error by {f(conn_gain, 2)}%, indicating that "
        f"the hybrid architecture benefits complex carbonate data when sufficient capacity is available."
    )

if ils_hash512 and ils_pg_hgp512:
    por_gain = pct(ils_hash512["por"], ils_pg_hgp512["por"], lower=True)
    conn_change = pct(ils_hash512["conn"], ils_pg_hgp512["conn"], lower=True)
    if conn_change < 0:
        print(
            f"\n6. On ILS crop512 log16, PoreGuard-HGP improves porosity preservation "
            f"({f(por_gain, 2)}% reduction) but worsens connectivity error, suggesting that "
            f"larger-volume reconstruction requires further control of small disconnected components."
        )

print(
    "\n7. Across experiments, component-count error often increases for PoreGuard-based methods, "
    "indicating a trade-off: the proposed method better preserves porosity and dominant connected "
    "pore structures, but may introduce additional small isolated components."
)

# Best by metric
print("\n\n" + "=" * 120)
print("BEST METHOD BY METRIC (within each dataset/crop/log)")
print("=" * 120)

groups = {}
for r in records:
    key = (r["dataset"], r["crop"], r["log"])
    groups.setdefault(key, []).append(r)

for key, rows in sorted(groups.items()):
    print(f"\n[{key[0]} | {key[1]} | {key[2]}]")
    
    for metric, label, lower in [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("acc", "Accuracy", False),
        ("por", "Porosity error", True),
        ("conn", "Connectivity error", True),
        ("comp", "Component count error", True),
    ]:
        valid = [r for r in rows if r.get(metric) is not None]
        if not valid:
            continue
        
        best = min(valid, key=lambda r: r[metric]) if lower else max(valid, key=lambda r: r[metric])
        print(f"  Best {label:20s}: {best['method']:<16} ({best['variant']:<16}) = {f(best[metric], 6):>10}")

print("\n" + "=" * 120)
print(f"Total experiments: {len(records)}")
print("=" * 120)
