import json
from pathlib import Path
import numpy as np
from skimage.measure import euler_number
from scipy.ndimage import label
root=Path('categorical_revision')
rows=[]
for sample in ['ILS','MEC']:
 gt=np.load(sample+'_crop256.npy'); labels=np.array([1,2,3]); truth=gt.reshape(-1)
 for seed in [2026,2027,2028]:
  for loss in ['mse','cross_entropy']:
   folder=root/f'{sample}_{loss}_seed{seed}'
   result=json.loads((folder/'results.json').read_text())
   raw=np.load(folder/'final_reconstruction.npy')
   pred=labels[np.abs(raw[...,None]*3-labels).argmin(-1)].astype(np.uint8)
   cm=np.bincount((truth.astype(np.int64)-1)*3+pred.reshape(-1)-1,minlength=9).reshape(3,3)
   denom=cm.sum(0)+cm.sum(1)-np.diag(cm)
   row={'sample':sample,'seed':seed,'loss':loss,'confusion_matrix':cm.tolist(),'accuracy':float(np.trace(cm)/cm.sum()),'mean_iou':float(np.mean(np.diag(cm)/denom)),'model_size_mb':result['model_size_mb'],'training_time_seconds':result['training_time_seconds'],'phases':{}}
   for phase in labels:
    a=gt==phase; b=pred==phase
    ea=int(euler_number(a,connectivity=3)); eb=int(euler_number(b,connectivity=3))
    _,na=label(a,structure=np.ones((3,3,3))); _,nb=label(b,structure=np.ones((3,3,3)))
    row['phases'][str(phase)]={'iou':float(cm[phase-1,phase-1]/denom[phase-1]),'fraction_gt':float(a.mean()),'fraction_pred':float(b.mean()),'euler_gt':ea,'euler_pred':eb,'euler_abs_error':abs(ea-eb),'components_gt':int(na),'components_pred':int(nb)}
   row['existing_metrics']=result['val_metrics'][-1]
   (folder/'categorical_metrics.json').write_text(json.dumps(row,indent=2)); rows.append(row)
(root/'all_metrics.json').write_text(json.dumps(rows,indent=2))
summary=[]
for sample in ['ILS','MEC']:
 for loss in ['mse','cross_entropy']:
  selected=[r for r in rows if r['sample']==sample and r['loss']==loss]
  item={'sample':sample,'loss':loss,'n':len(selected)}
  for key in ['accuracy','mean_iou','model_size_mb','training_time_seconds']:
   vals=[r[key] for r in selected]; item[key]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1))}
  summary.append(item)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print('ALL_ANALYSIS_COMPLETE')
