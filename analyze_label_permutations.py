import itertools, json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from skimage.measure import euler_number

ROOT = Path(__file__).parent
OUT = ROOT / 'label_permutation_revision'
ORIG = ROOT / 'server_backup' / 'extracted'
rows = []

for sample in ('ILS', 'MEC'):
    gt = np.load(ORIG / f'{sample}_crop256.npy').astype(np.int16)
    for perm in itertools.permutations((1, 2, 3)):
        code = ''.join(map(str, perm)); d = OUT / f'{sample}_perm{code}'
        result = json.loads((d / 'results.json').read_text())
        raw = np.load(d / 'final_reconstruction.npy')
        # Input label p(original label) = perm[index], so undo the permutation here.
        pred_perm = np.rint(raw * 3).clip(1, 3).astype(np.int16)
        inv = {int(v): int(k) + 1 for k, v in enumerate(perm)}
        pred = np.zeros_like(pred_perm)
        for p, original in inv.items(): pred[pred_perm == p] = original
        y = gt.ravel(); z = pred.ravel(); cm = confusion_matrix(y, z, labels=[1, 2, 3])
        iou = np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1) - np.diag(cm))
        pore_gt = np.isin(gt, [1, 3]); pore_pred = np.isin(pred, [1, 3])
        def topo(mask):
            lab, n = ndimage.label(mask, structure=np.ones((3,3,3)))
            sizes = np.bincount(lab.ravel())[1:]
            return n, (sizes.max() / mask.sum() if sizes.size and mask.sum() else 0.0), int(euler_number(mask, connectivity=3))
        ng, lg, eg = topo(pore_gt); npred, lp, ep = topo(pore_pred)
        rows.append({'sample': sample, 'permutation': code, 'accuracy': float(np.mean(y == z)),
                     'mean_iou': float(iou.mean()), 'iou': iou.tolist(),
                     'porosity_error': float(abs(pore_gt.mean() - pore_pred.mean())),
                     'connectivity_error': float(abs(lg - lp)), 'component_count_error': abs(npred-ng),
                     'euler_abs_error': abs(ep-eg), 'confusion_matrix': cm.tolist(),
                     'training_time_seconds': result['training_time_seconds']})
(OUT / 'label_permutation_metrics.json').write_text(json.dumps(rows, indent=2))
for sample in ('ILS', 'MEC'):
    ss = [r for r in rows if r['sample'] == sample]
    print(sample)
    for k in ('accuracy','mean_iou','porosity_error','connectivity_error','component_count_error','euler_abs_error'):
        v = np.array([r[k] for r in ss], float)
        print(f'  {k}: mean={v.mean():.6f} sd={v.std(ddof=1):.6f} min={v.min():.6f} max={v.max():.6f}')
