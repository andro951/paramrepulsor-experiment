#!/usr/bin/env python3
"""Predeclared 10 matched seeds (0..9), all three existing datasets, all outcomes.
Four isolated single-thread CPU workers; wall times are descriptive, not a benchmark.
"""
import concurrent.futures,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def run(job):
 n,seed,variant=job; name=f'{variant}_n{n}_seed{seed}'; out=ROOT/'experiment/results'/name
 with open(str(out)+'.log','w') as f:
  subprocess.run([sys.executable,str(ROOT/'experiment/run_trial.py'),'--variant',variant,'--n',str(n),'--seed',str(seed)],check=True,stdout=f,stderr=subprocess.STDOUT)
 meta=json.loads(pathlib.Path(str(out)+'.json').read_text());return name,meta['seconds'],meta['metrics']
jobs=[(n,seed,variant) for n in (36,50,100) for seed in range(10) for variant in ('upstream','modified')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(run,jobs):print(*result,flush=True)
