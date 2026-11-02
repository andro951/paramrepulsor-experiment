#!/usr/bin/env python3
"""Sequential, warmed pair-generation timing; not a large-N performance claim."""
import argparse,json,os,pathlib,sys,time
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['upstream','modified'],required=True);a=p.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['TORCH_DEVICE']='cpu';os.environ['NUMBA_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.path.insert(0,str(ROOT/a.variant/'src'))
import numpy as np,pandas as pd
from parampacmap.utils.data import generate_pair
rows=[]
for n in (36,50,100):
 X=pd.read_csv(ROOT/'inputs/datasets'/f'paramrep_2d_{n}.csv')[['math_score','english_score']].to_numpy(np.float32)
 generate_pair(X,10,5,20,verbose=False,random_state=0) # warm all compiled paths
 for seed in range(10):
  t=time.perf_counter();generate_pair(X,10,5,20,verbose=False,random_state=seed);elapsed=time.perf_counter()-t
  rows.append({'variant':a.variant,'n':n,'seed':seed,'warmed_pair_seconds':elapsed})
print(json.dumps(rows,indent=2))
