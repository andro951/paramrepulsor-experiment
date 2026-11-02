#!/usr/bin/env python3
"""Run one isolated baseline/modified trial; training algorithm is unmodified."""
import argparse, hashlib, json, os, pathlib, sys, time, warnings
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--variant',choices=['upstream','modified'],required=True); p.add_argument('--n',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--output'); a=p.parse_args()
os.environ['TORCH_DEVICE']='cpu'; os.environ.setdefault('NUMBA_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/a.variant/'src'))
import numpy as np, pandas as pd, torch
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from parampacmap import ParamPaCMAP
from parampacmap.utils import data
warnings.filterwarnings('ignore',message=".*pin_memory.*")
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
f=ROOT/'inputs/datasets'/f'paramrep_2d_{a.n}.csv'; df=pd.read_csv(f); X=df[['math_score','english_score']].to_numpy(dtype=np.float32); labels=df.group.to_numpy(); ids=df.point_id.to_numpy(dtype=str)
class Observed(ParamPaCMAP):
    """Read-only, RNG-free epoch observations; no changes to training or optimizer."""
    def _train_epoch(self, train_loader, epoch, optimizer, has_labels=False):
        if epoch==0:
            self.initial_hash=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in self.model.state_dict().values())).hexdigest()
            self.initial_rng=torch.get_rng_state().clone()
            self.epochs=[0]; self.frames=[self.observe()]
        super()._train_epoch(train_loader,epoch,optimizer,has_labels)
        if (epoch+1)%5==0 or epoch==0:
            before=torch.get_rng_state().clone(); self.epochs.append(epoch+1); self.frames.append(self.observe()); assert torch.equal(before,torch.get_rng_state())
    def _inference(self, test_loader):
        # Cloud sandbox blocks PyTorch IPC sockets; identical in-process inference
        # over this <=100-row single batch avoids worker transport only.
        return self.observe()
    def observe(self):
        with torch.inference_mode():
            return self.model(torch.as_tensor(X,dtype=self._dtype,device=self.device)).detach().cpu().numpy().copy()
r=Observed(seed=a.seed,save_pairs=True) # all numerical/model/loss/training defaults untouched
start=time.perf_counter(); Y=r.fit_transform(X); seconds=time.perf_counter()-start
NN=r.pair_neighbors[:,1].reshape(a.n,-1); MN=r.pair_MN[:,1].reshape(a.n,-1); FP=r.pair_FP[:,1].reshape(a.n,-1)
def audit(P):
    return {'self_pairs':int(np.sum(P==np.arange(a.n)[:,None])), 'duplicate_excess':int(sum(len(row)-len(set(row)) for row in P)), 'nn_overlap':int(sum(np.isin(P[i],NN[i]).sum() for i in range(a.n))),'same_group':int(sum((labels[row]==labels[i]).sum() for i,row in enumerate(P))),'total':int(P.size)}
def neighbors(Z,k=10):
    d=cdist(Z,Z); np.fill_diagonal(d,np.inf); return np.argsort(d,axis=1,kind='stable')[:,:k]
xnbr=neighbors(X); ynbr=neighbors(Y)
# Orthogonal similarity alignment: fitted translation, rotation/reflection and uniform scale.
# Embedding orientation has no physical meaning; raw outputs are also preserved.
x0=X-X.mean(0); y0=Y-Y.mean(0); u,s,vt=np.linalg.svd(y0.T@x0); rot=u@vt; scale=float(np.sum(s)/np.sum(y0*y0)); aligned=y0@rot*scale+X.mean(0)
metrics={'silhouette':float(silhouette_score(Y,labels)),'knn10_group_purity':float(np.mean(labels[ynbr]==labels[:,None])),'knn10_recall':float(np.mean([len(set(xnbr[i])&set(ynbr[i]))/10 for i in range(a.n)])),'trustworthiness10':float(trustworthiness(X,Y,n_neighbors=10)),'distance_spearman':float(spearmanr(pdist(X),pdist(Y)).statistic),'aligned_rms_score_units':float(np.sqrt(np.mean(np.sum((aligned-X)**2,axis=1))))}
meta={'variant':a.variant,'n':a.n,'seed':a.seed,'seconds':seconds,'initial_weights_sha256':r.initial_hash,'metrics':metrics,'audit':{'NN':audit(NN),'MN':audit(MN),'FP':audit(FP),'mn_fp_overlap':int(sum(np.isin(MN[i],FP[i]).sum() for i in range(a.n)))},'alignment':{'rotation':rot.tolist(),'scale':scale,'input_mean':X.mean(0).tolist(),'output_mean':Y.mean(0).tolist(),'reflection_allowed':True,'reflection_used':bool(np.linalg.det(rot)<0)},'defaults':{'n_neighbors':r.n_neighbors,'n_MN':r.n_MN,'n_FP':r.n_FP,'num_epochs':r.num_epochs,'lr':r.lr,'batch_size':r.batch_size,'model_dict':r.model_dict,'apply_scale':r.apply_scale,'apply_pca':r.apply_pca,'optim_type':r.optim_type,'weight_schedule':r.weight_schedule.__name__,'const_schedule':r.const_schedule.__name__,'num_workers':r.num_workers,'dtype':str(r._dtype)}}
out=pathlib.Path(a.output) if a.output else ROOT/'experiment/results'/f'{a.variant}_n{a.n}_seed{a.seed}'; out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(str(out)+'.npz',X=X,Y=Y,aligned=aligned,NN=NN,MN=MN,FP=FP,ids=ids,labels=labels.astype(str),epochs=np.array(r.epochs),frames=np.stack(r.frames),initial_rng=r.initial_rng.numpy())
pathlib.Path(str(out)+'.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta),flush=True)
