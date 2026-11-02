#!/usr/bin/env python3
"""Analyze every predeclared run and draw static numerical plots (Matplotlib)."""
import json,pathlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=pathlib.Path(__file__).resolve().parents[1]; R=ROOT/'experiment/results'; P=ROOT/'experiment/plots'; P.mkdir(exist_ok=True)
COLORS={'High Both':'#009E73','Low Both':'#E69F00','Math Strong':'#CC79A7','English Strong':'#0072B2'}
rows=[];matches=[]
for n in (36,50,100):
 for seed in range(10):
  b=np.load(R/f'upstream_n{n}_seed{seed}.npz');m=np.load(R/f'modified_n{n}_seed{seed}.npz')
  bm=json.loads((R/f'upstream_n{n}_seed{seed}.json').read_text());mm=json.loads((R/f'modified_n{n}_seed{seed}.json').read_text())
  for key in ('X','NN','ids','labels','initial_rng'):
   assert np.array_equal(b[key],m[key]),f'Unmatched {key} {n} {seed}'
  assert bm['initial_weights_sha256']==mm['initial_weights_sha256']
  assert np.array_equal(b['frames'][0],m['frames'][0])
  matches.append({'n':n,'seed':seed,'same_NN':True,'same_initial_weights':True,'same_initial_embedding':True,'same_initial_training_rng':True})
  for variant,meta in [('upstream',bm),('modified',mm)]:
   # Refit all alignments at analysis time, including early-run artifacts made
   # before this visualization choice was finalized. Raw X and Y never change.
   f=R/f'{variant}_n{n}_seed{seed}.npz'; z=dict(np.load(f)); X=z['X'].astype(float);Y=z['Y'].astype(float);x0=X-X.mean(0);y0=Y-Y.mean(0);u,s,vt=np.linalg.svd(y0.T@x0);rot=u@vt;scale=s.sum()/np.sum(y0*y0);z['aligned']=y0@rot*scale+X.mean(0);np.savez_compressed(f,**z)
   meta['metrics']['aligned_rms_score_units']=float(np.sqrt(np.mean(np.sum((z['aligned']-X)**2,axis=1))));meta['alignment']={'rotation_or_reflection':rot.tolist(),'scale':float(scale),'input_mean':X.mean(0).tolist(),'output_mean':Y.mean(0).tolist(),'reflection_allowed':True,'reflection_used':bool(np.linalg.det(rot)<0)};(R/f'{variant}_n{n}_seed{seed}.json').write_text(json.dumps(meta,indent=2))
   row={'n':n,'seed':seed,'variant':variant,'seconds':meta['seconds'],**meta['metrics']}
   for family in ('MN','FP'):
    for k,v in meta['audit'][family].items():row[f'{family}_{k}']=v
   row['mn_fp_overlap']=meta['audit']['mn_fp_overlap'];rows.append(row)
df=pd.DataFrame(rows);df.to_csv(R/'all_trials.csv',index=False);pd.DataFrame(matches).to_csv(R/'matched_verification.csv',index=False)
metrics=['silhouette','knn10_group_purity','knn10_recall','trustworthiness10','distance_spearman','aligned_rms_score_units','seconds']
summary=df.groupby(['n','variant'])[metrics].agg(['mean','std','min','max']);summary.to_csv(R/'summary.csv')
paired=[]
for n in (36,50,100):
 for metric in metrics:
  pivot=df[df.n==n].pivot(index='seed',columns='variant',values=metric);delta=pivot.modified-pivot.upstream
  paired.append({'n':n,'metric':metric,'baseline_mean':pivot.upstream.mean(),'modified_mean':pivot.modified.mean(),'delta_mean':delta.mean(),'delta_std':delta.std(),'delta_min':delta.min(),'delta_max':delta.max(),'positive_deltas':int((delta>1e-12).sum()),'negative_deltas':int((delta< -1e-12).sum()),'zero_deltas':int((delta.abs()<=1e-12).sum()),'zero_tolerance':1e-12})
pd.DataFrame(paired).to_csv(R/'paired_differences.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def scatter(ax,Z,labels,ids=None,alpha=1,s=35):
 for label,color in COLORS.items():
  mask=labels==label
  if np.any(mask):ax.scatter(Z[mask,0],Z[mask,1],c=color,s=s,label=f'{label} (n={mask.sum()})',alpha=alpha,edgecolors='white',linewidths=.4,zorder=3)
 if ids is not None:
  for xy,id in zip(Z,ids):ax.annotate(id,xy,xytext=(3,2),textcoords='offset points',fontsize=5.8,alpha=.9).set_gid('point-id')
 ax.set_aspect('equal',adjustable='datalim');ax.margins(.15);ax.grid(alpha=.15)
def save(fig,name):
 point_labels=[obj for obj in fig.findobj() if obj.get_gid()=='point-id']
 if point_labels:fig.savefig(P/f'{name}_point_ids.svg',bbox_inches='tight')
 for obj in point_labels:obj.set_visible(False)
 fig.savefig(P/f'{name}.png',dpi=200,bbox_inches='tight');fig.savefig(P/f'{name}.svg',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(15,14),constrained_layout=True)
for row,n in enumerate((36,50,100)):
 b=np.load(R/f'upstream_n{n}_seed0.npz');m=np.load(R/f'modified_n{n}_seed0.npz')
 for col,(Z,title) in enumerate(((b['X'],'Input scores'),(b['Y'],'Released baseline: raw output'),(m['Y'],'Requested sampler: raw output'))):
  ax=axs[row,col];scatter(ax,Z,b['labels']);ax.set_title(f'{n} points · {title}');ax.set_xlabel('Math score' if col==0 else 'Embedding coordinate 1');ax.set_ylabel('English score' if col==0 else 'Embedding coordinate 2')
 axs[row,0].legend(fontsize=8,loc='best')
fig.suptitle('ParamRepulsor · same data, same initial model, same seed (0)\nRaw outputs have arbitrary translation, orientation and scale; no outcome-based seed selection.',fontsize=14)
save(fig,'comparison_overview_seed0')
for n in (36,50,100):
 b=np.load(R/f'upstream_n{n}_seed0.npz');m=np.load(R/f'modified_n{n}_seed0.npz');X=b['X'];labels=b['labels'];ids=b['ids']
 fig,axs=plt.subplots(1,3,figsize=(19,6),constrained_layout=True)
 for col,(Z,title) in enumerate(((X,'Original input'),(b['Y'],'Released baseline · raw output'),(m['Y'],'Requested sampler · raw output'))):
  scatter(axs[col],Z,labels,ids);axs[col].set_title(title);axs[col].set_xlabel('Math score' if col==0 else 'Embedding coordinate 1');axs[col].set_ylabel('English score' if col==0 else 'Embedding coordinate 2')
 axs[0].legend(fontsize=8);fig.suptitle(f'{n} points · 450 real training epochs · fixed seed 0 · stable cluster colors (point-ID overlay in companion SVG)',fontsize=14);save(fig,f'input_output_{n}_seed0')
 fig,axs=plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
 allz=np.vstack([X,b['aligned'],m['aligned']]);lo=allz.min(0)-4;hi=allz.max(0)+4
 for ax,z,title in ((axs[0],b,'Released baseline'),(axs[1],m,'Requested sampler')):
  Y=z['aligned']
  for i,label in enumerate(labels):
   c=COLORS[str(label)];ax.scatter(*X[i],color=c,s=32,alpha=.20,zorder=2);ax.annotate('',xy=Y[i],xytext=X[i],arrowprops={'arrowstyle':'->','color':c,'alpha':.5,'lw':.75,'shrinkA':0,'shrinkB':2},zorder=1)
  scatter(ax,Y,labels,ids,s=40);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('Aligned coordinate 1 (input-score units)');ax.set_ylabel('Aligned coordinate 2 (input-score units)')
 axs[0].legend(fontsize=8);fig.suptitle(f'{n} points · input → final aligned embedding · seed 0\nFaint dots = input; solid dots = final. Arrows show correspondence, NOT actual training paths.',fontsize=13)
 fig.supxlabel('Output fit to input using one global translation, rotation/reflection and uniform scale; raw outputs shown separately.',fontsize=10)
 save(fig,f'movement_aligned_{n}_seed0')
# Every paired result, not a selected seed.
fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,metric,title in zip(axs.flat,metrics[:6],['Cluster silhouette ↑','10-NN group purity ↑','Original 10-NN recall ↑','Trustworthiness (10) ↑','Pairwise distance rank correlation ↑','Aligned RMS (score units) ↓']):
 for j,n in enumerate((36,50,100)):
  sub=df[df.n==n].pivot(index='seed',columns='variant',values=metric)
  for _,row in sub.iterrows():ax.plot([j-.16,j+.16],[row.upstream,row.modified],color='#777777',alpha=.35,lw=.8)
  ax.scatter(np.full(10,j-.16),sub.upstream,color='#444444',s=22,label='Baseline' if j==0 else None,zorder=3)
  ax.scatter(np.full(10,j+.16),sub.modified,color='#D55E00',s=22,label='Modified' if j==0 else None,zorder=3)
 ax.set_xticks(range(3),['36 points','50 points','100 points']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
 if metric=='knn10_group_purity':ax.set_ylim(0,1.05);ax.set_yticks([0,.25,.5,.75,1])
axs.flat[0].legend();fig.suptitle('All 10 matched seeds (0–9), all three datasets · each line is one seed\nCluster separation and fine-neighbor preservation are different objectives.',fontsize=14);save(fig,'all_seeds_metrics')
print(pd.DataFrame(paired).to_string(index=False));print('All 30 pairs match NN, initialization, inputs and initial training RNG; plots saved.')
