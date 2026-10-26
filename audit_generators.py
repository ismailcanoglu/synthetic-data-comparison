import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'D4_extension'))
from experiment_d1 import sample_dgp
from experiment_d2_d3 import generate
seeds=range(30)
rows=[]
for s in seeds:
 x1,y1=sample_dgp(1000,np.random.default_rng(s))
 x3,y3=generate(1000,np.random.default_rng(s),'d3')
 x2,y2=generate(1000,np.random.default_rng(s),'d2')
 xt,yt=generate(2000,np.random.default_rng(s+100000),'d3',shift=True)
 rows.append((y1.sum(),y2.sum(),np.array_equal(x1,x3),np.array_equal(y1,y3),np.mean(x3[:,5]),np.mean(xt[:,5])))
a=np.array(rows)
print('D1 D3 exact match for 30 seeds:',bool(a[:,2].all() and a[:,3].all()))
print('D1 minority count n1000 min mean max',a[:,0].min(),a[:,0].mean(),a[:,0].max())
print('D2 minority count n1000 min mean max',a[:,1].min(),a[:,1].mean(),a[:,1].max())
print('D3 sine feature mean train/test across seeds',a[:,4].mean(),a[:,5].mean())
for name in ['d1','d2','d3']:
 x,y=(sample_dgp(10000,np.random.default_rng(7)) if name=='d1' else generate(10000,np.random.default_rng(7),name))
 print(name,'minority proportion',y.mean(),'feature means class0',np.round(x[y==0].mean(axis=0),3),'class1',np.round(x[y==1].mean(axis=0),3))
