#!/usr/bin/env python3
"""Final numerical/artifact invariants for the exact full experiment."""
import json,pathlib,subprocess
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];R=ROOT/'experiment/results'
df=pd.read_csv(R/'all_trials.csv');assert len(df)==60;assert len(df[['n','seed','variant']].drop_duplicates())==60
assert set(df.seed)==set(range(10));assert set(df.n)=={36,50,100}
for n in (36,50,100):
 for seed in range(10):
  for variant in ('upstream','modified'):
   z=np.load(R/f'{variant}_n{n}_seed{seed}.npz');meta=json.loads((R/f'{variant}_n{n}_seed{seed}.json').read_text());assert z['X'].shape==z['Y'].shape==(n,2);assert np.isfinite(z['Y']).all();assert np.isfinite(z['aligned']).all();assert z['epochs'][0]==0 and z['epochs'][-1]==450;assert np.array_equal(z['frames'][-1],z['Y']);assert len(set(z['ids']))==n
   assert meta['defaults']['num_epochs']==450 and meta['defaults']['n_neighbors']==10 and meta['defaults']['n_MN']==5 and meta['defaults']['n_FP']==20
   if variant=='modified':
    for family in ('MN','FP'):
     assert all(v==0 for k,v in meta['audit'][family].items() if k in ('self_pairs','duplicate_excess','nn_overlap'))
    # Overlap is not excluded and is actually present in these default runs.
    assert meta['audit']['mn_fp_overlap']>0
print('PASS all60 trials finite, exact defaults, actual450-epoch finals; modified exclusions/uniqueness; allowed MN/FP overlap')
# Nested exact data and labels remain untouched.
base=pd.read_csv(ROOT/'inputs/datasets/paramrep_2d_100.csv').set_index('point_id')
for n in (36,50):
 sub=pd.read_csv(ROOT/'inputs/datasets'/f'paramrep_2d_{n}.csv').set_index('point_id');assert sub.equals(base.loc[sub.index])
print('PASS exact nested input coordinates/labels/IDs')
if (ROOT/'upstream/.git').exists():
 assert not subprocess.check_output(['git','-C',str(ROOT/'upstream'),'status','--porcelain']).strip()
else:
 import hashlib
 expected=json.loads((ROOT/'experiment/provenance.json').read_text())['baseline_tracked_file_sha256']
 assert all(hashlib.sha256((ROOT/'upstream'/name).read_bytes()).hexdigest()==sha for name,sha in expected.items())
print('PASS untouched baseline source')
