"""Check observer is RNG-free and training-equivalent to unobserved upstream fit.
Only inference multiprocessing transport is replaced, identically in both arms.
"""
import hashlib,os,pathlib,sys
os.environ['TORCH_DEVICE']='cpu';os.environ['NUMBA_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'upstream/src'))
import numpy as np,pandas as pd,torch
from parampacmap import ParamPaCMAP
X=pd.read_csv(ROOT/'inputs/datasets/paramrep_2d_36.csv')[['math_score','english_score']].to_numpy(np.float32)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
class InProcessInference(ParamPaCMAP):
 def _inference(self,loader):
  # Reproduce the original single batch calculation without worker transport.
  # This call happens after all training epochs (snapshots disabled by default).
  with torch.inference_mode():return self.model(torch.as_tensor(X,dtype=self._dtype,device=self.device)).detach().cpu().numpy()
r=InProcessInference(seed=0,save_pairs=True);Y=r.fit_transform(X)
b=np.load(ROOT/'experiment/results/upstream_n36_seed0.npz')
for name,actual in [('Y',Y),('NN',r.pair_neighbors[:,1].reshape(len(X),-1)),('MN',r.pair_MN[:,1].reshape(len(X),-1)),('FP',r.pair_FP[:,1].reshape(len(X),-1))]:
 assert np.array_equal(actual,b[name]),name
 print('PASS exact equality observed vs no-observer:',name)
# Test the unmodified _inference calculation with its identical dataset/batch,
# using 0 workers only to avoid the sandbox-denied IPC transport.
from parampacmap.models.dataset import TensorDataset
loader=torch.utils.data.DataLoader(TensorDataset(X),batch_size=2048,shuffle=False,num_workers=0)
original=ParamPaCMAP._inference(r,loader)
assert np.array_equal(original,Y)
print('PASS exact equality original inference arithmetic vs direct-model adapter')
