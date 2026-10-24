"""Voxel pore-network transport proxy for paired reconstruction comparison.

This is a dimensionless relative permeability proxy, not an absolute Darcy
solver. Pore voxels are unit-conductance graph nodes and pressure is solved
with fixed inlet/outlet values and no-flow solid boundaries.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / "server_backup" / "extracted"
OUT = ROOT / "permeability_revision"
SAMPLES = {"ILS": BACKUP / "ILS_crop256.npy", "MEC": BACKUP / "MEC_crop256.npy"}


def majority_downsample(a, factor=4):
    d, h, w = a.shape
    a = a[: d // factor * factor, : h // factor * factor, : w // factor * factor]
    b = a.reshape(d // factor, factor, h // factor, factor, w // factor, factor)
    # For a three-phase segmentation, pore is labels 1 and 3. Majority vote
    # among the original labels preserves solid where it dominates a block.
    vals = np.stack([(b == v).sum(axis=(1, 3, 5)) for v in (1, 2, 3)], axis=0)
    return np.array((1, 2, 3), dtype=np.uint8)[vals.argmax(axis=0)]


def solve_axis(pore, axis):
    """Return conductance proxy for unit pressure drop along axis."""
    pore = np.asarray(pore, dtype=bool)
    shape = pore.shape
    coords = np.argwhere(pore)
    if len(coords) == 0:
        return 0.0, {"nodes": 0, "inlet_nodes": 0, "outlet_nodes": 0}
    ids = -np.ones(shape, dtype=np.int64)
    ids[pore] = np.arange(len(coords))
    inlet = coords[:, axis] == 0
    outlet = coords[:, axis] == shape[axis] - 1
    active = ~(inlet | outlet)
    interior = np.flatnonzero(active)
    if not inlet.any() or not outlet.any() or len(interior) == 0:
        return 0.0, {"nodes": int(len(coords)), "inlet_nodes": int(inlet.sum()), "outlet_nodes": int(outlet.sum())}
    row = {int(g): i for i, g in enumerate(interior)}
    rr, cc, dd = [], [], []
    rhs = np.zeros(len(interior), dtype=np.float64)
    dirs = []
    for ax in range(3):
        for sign in (-1, 1):
            v = np.zeros(3, dtype=int); v[ax] = sign; dirs.append(v)
    for g in interior:
        i = row[int(g)]; c = coords[g]; degree = 0
        for dv in dirs:
            q = c + dv
            if np.any(q < 0) or np.any(q >= shape) or not pore[tuple(q)]:
                continue
            degree += 1; ng = int(ids[tuple(q)])
            rr.append(i); cc.append(i); dd.append(1.0)
            if inlet[ng]:
                rhs[i] += 1.0
            elif not outlet[ng]:
                rr.append(i); cc.append(row[ng]); dd.append(-1.0)
        if degree == 0:
            rr.append(i); cc.append(i); dd.append(1.0)
    A = sparse.csr_matrix((dd, (rr, cc)), shape=(len(interior), len(interior)))
    p, info = cg(A, rhs, rtol=1e-7, atol=1e-10, maxiter=3000)
    if info != 0:
        raise RuntimeError(f"CG failed axis={axis}, info={info}")
    pressure = np.zeros(len(coords), dtype=np.float64)
    pressure[inlet] = 1.0
    pressure[outlet] = 0.0
    pressure[interior] = p
    flux = 0.0
    for g in np.flatnonzero(inlet):
        c = coords[g]
        q = c.copy(); q[axis] += 1
        if q[axis] < shape[axis] and pore[tuple(q)]:
            flux += pressure[g] - pressure[ids[tuple(q)]]
    meta = {"nodes": int(len(coords)), "inlet_nodes": int(inlet.sum()), "outlet_nodes": int(outlet.sum()), "cg_info": int(info)}
    return float(max(flux, 0.0)), meta


def permeability(a):
    pore = np.isin(a, (1, 3))
    values, meta = [], {}
    for ax in range(3):
        value, details = solve_axis(pore, ax)
        values.append(value)
        meta[str(ax)] = details
    return np.asarray(values), meta


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for sample, gt_path in SAMPLES.items():
        gt = majority_downsample(np.load(gt_path))
        cases = [("ground_truth", gt)]
        for loss in ("mse", "cross_entropy"):
            for seed in (2026, 2027, 2028):
                p = BACKUP / "categorical_revision" / f"{sample}_{loss}_seed{seed}" / "final_reconstruction.npy"
                raw = np.load(p)
                pred = np.array((1, 2, 3), dtype=np.uint8)[np.abs(raw[..., None] * 3 - np.array((1, 2, 3))).argmin(-1)]
                cases.append((f"{loss}_seed{seed}", majority_downsample(pred)))
        gt_k, gt_meta = permeability(gt)
        for name, volume in cases:
            k, meta = permeability(volume)
            row = {"sample": sample, "case": name, "grid_shape": list(volume.shape), "axes": ["x", "y", "z"], "proxy": k.tolist(), "relative_to_ground_truth": (k / np.maximum(gt_k, 1e-12)).tolist(), "metadata": meta}
            rows.append(row)
    (OUT / "permeability_results.json").write_text(json.dumps(rows, indent=2))
    summary = {}
    for sample in SAMPLES:
        gt = next(r for r in rows if r["sample"] == sample and r["case"] == "ground_truth")
        summary[sample] = {"ground_truth_proxy": gt["proxy"]}
        for loss in ("mse", "cross_entropy"):
            rr = [r for r in rows if r["sample"] == sample and r["case"].startswith(loss)]
            vals = np.array([r["relative_to_ground_truth"] for r in rr])
            summary[sample][loss] = {"mean_relative": vals.mean(0).tolist(), "sd_relative": vals.std(0, ddof=1).tolist(), "mean_abs_relative_error": float(np.abs(vals - 1).mean())}
    (OUT / "permeability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
