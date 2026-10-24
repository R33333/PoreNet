import itertools, subprocess, json, os, sys
from pathlib import Path
root=Path(r'D:\code\inr_digital_rock'); out=root/'label_permutation_revision'; logs=out/'logs'; logs.mkdir(exist_ok=True)
common=['--loss_type','mse','--encoding_type','hash','--n_levels','16','--log2_hashmap_size','15','--hidden_dim','64','--n_hidden_layers','3','--batch_size','16384','--samples_per_epoch','1048576','--epochs','60','--validate_every','10','--seed','2026','--num_workers','0']
for sample in ['ILS','MEC']:
 for perm in itertools.permutations([1,2,3]):
  name=''.join(map(str,perm)); od=out/f'{sample}_perm{name}'
  if (od/'results.json').exists(): continue
  cmd=[sys.executable,'-u','train.py','--data',str(out/f'{sample}_perm{name}.npy'),'--output',str(od)]+common
  with open(logs/f'{sample}_perm{name}.log','w') as f:
   rc=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc: raise SystemExit(f'failed {sample} {name}')
print('PERMUTATION_TRAINING_COMPLETE')
