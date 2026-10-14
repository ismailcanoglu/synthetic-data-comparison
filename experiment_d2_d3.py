"""D2/D3 controlled data-generation study. Run: python experiment_d2_d3.py --epochs 100"""
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from imblearn.over_sampling import SMOTE

ROOT=Path(__file__).resolve().parent
SEEDS=[11,23,37,51,79]
METRICS=['f1_macro','f1_minority','balanced_accuracy','auroc','pr_auc']

def generate(n,rng,scenario,minority_only=False,shift=False):
    y=np.ones(n,dtype=int) if minority_only else rng.binomial(1,.1,size=n)
    z=rng.normal(size=(n,3))
    if scenario=='d2':
        # Minority forms a ring, majority near center: nonlinear Bayes boundary.
        angle=rng.uniform(0,2*np.pi,n)
        radius=np.where(y==1,2.3,.55)+rng.normal(0,.28,n)
        z[:,0]=radius*np.cos(angle);z[:,1]=radius*np.sin(angle)
        z[:,2]+=y*.4
    else:
        # D3: same label prior and class mechanism, shifted nuisance variable.
        z[:,0]+=1.1*y;z[:,1]-=.8*y
        z[:,2]+=(1.5 if shift else 0.)
    x=np.column_stack([z,z[:,0]*z[:,1],z[:,0]**2,np.sin(z[:,2])])
    x+=rng.normal(0,.08,size=x.shape)
    return x,y

def ctgan_samples(X,y,n,seed,epochs):
    from ctgan import CTGAN
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cols=[f'f{i}' for i in range(X.shape[1])]
    df=pd.DataFrame(X,columns=cols);df['target']=y
    model=CTGAN(epochs=epochs,batch_size=100,verbose=False,enable_gpu=False)
    model.fit(df,discrete_columns=['target'])
    parts=[];count=0
    for _ in range(30):
        batch=model.sample(max(200,(n-count)*2),condition_column='target',condition_value=1)
        batch['target']=pd.to_numeric(batch['target'],errors='coerce')
        for c in cols:batch[c]=pd.to_numeric(batch[c],errors='coerce')
        batch=batch.loc[batch.target.eq(1)].dropna(subset=cols)
        if len(batch):parts.append(batch[cols].to_numpy(float));count+=len(batch)
        if count>=n:break
    if count<n:raise RuntimeError(f'Only {count}/{n} CTGAN minority samples')
    return np.vstack(parts)[:n]

def mmd2(a,b):
    # Median heuristic bandwidth from pooled subset, biased MMD^2 estimator.
    rng=np.random.default_rng(2026)
    a=a[rng.choice(len(a),min(250,len(a)),replace=False)]
    b=b[rng.choice(len(b),min(250,len(b)),replace=False)]
    z=np.vstack([a,b]);dist=((z[:,None,:]-z[None,:,:])**2).sum(-1)
    positive=dist[dist>0];gamma=1/(np.median(positive)+1e-12) if len(positive) else 1.
    return float(rbf_kernel(a,a,gamma=gamma).mean()+rbf_kernel(b,b,gamma=gamma).mean()-2*rbf_kernel(a,b,gamma=gamma).mean())

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--epochs',type=int,default=100);p.add_argument('--skip-ctgan',action='store_true');args=p.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];quality=[];fail=[]
    for scenario in ('d2','d3'):
        for seed in args.seeds:
            Xtr,ytr=generate(1000,np.random.default_rng(seed),scenario)
            Xte,yte=generate(2000,np.random.default_rng(seed+100000),scenario,shift=scenario=='d3')
            n_add=int((ytr==0).sum()-(ytr==1).sum())
            Xsm,ysm=SMOTE(random_state=seed,k_neighbors=5).fit_resample(Xtr,ytr)
            Xor,_=generate(n_add,np.random.default_rng(seed+200000),scenario,minority_only=True)
            methods={'baseline':(Xtr,ytr,None),'smote':(Xsm,ysm,Xsm[len(Xtr):]),'oracle_simulation':(np.vstack([Xtr,Xor]),np.r_[ytr,np.ones(n_add,dtype=int)],Xor)}
            if not args.skip_ctgan:
                try:
                    Xct=ctgan_samples(Xtr,ytr,n_add,seed,args.epochs)
                    methods['ctgan']=(np.vstack([Xtr,Xct]),np.r_[ytr,np.ones(n_add,dtype=int)],Xct)
                except Exception as exc:
                    fail.append({'scenario':scenario,'seed':seed,'method':'ctgan','error':str(exc)})
                    print('CTGAN failed:',scenario,seed,exc,flush=True)
            # Independent oracle reference for fidelity; never used in model training.
            ref,_=generate(1000,np.random.default_rng(seed+300000),scenario,minority_only=True)
            for method,(X,y,added) in methods.items():
                if added is not None:
                    quality.append({'scenario':scenario,'seed':seed,'method':method,'mmd2_oracle_minority':mmd2(added,ref),'n_added':len(added)})
                classifiers={'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed)), 'random_forest':RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=-1)}
                for name,clf in classifiers.items():
                    clf.fit(X,y);pred=clf.predict(Xte);proba=clf.predict_proba(Xte)[:,1]
                    rows.append({'scenario':scenario,'seed':seed,'method':method,'classifier':name,'n_train':len(X),'n_added':len(X)-len(Xtr),'f1_macro':f1_score(yte,pred,average='macro'),'f1_minority':f1_score(yte,pred,pos_label=1,zero_division=0),'balanced_accuracy':balanced_accuracy_score(yte,pred),'auroc':roc_auc_score(yte,proba),'pr_auc':average_precision_score(yte,proba)})
            pd.DataFrame(rows).to_csv(out/'d2_d3_per_seed.csv',index=False)
            pd.DataFrame(quality).to_csv(out/'d2_d3_quality.csv',index=False)
            pd.DataFrame(fail,columns=['scenario','seed','method','error']).to_csv(out/'d2_d3_failures.csv',index=False)
            print('Completed',scenario,'seed',seed,flush=True)
    df=pd.DataFrame(rows)
    df.groupby(['scenario','method','classifier'])[METRICS].agg(['mean','std']).to_csv(out/'d2_d3_summary.csv')
    pairs=[]
    for (scenario,clf),sub in df.groupby(['scenario','classifier']):
        baseline=sub[sub.method=='baseline'].set_index('seed')
        for method in sorted(set(sub.method)-{'baseline'}):
            other=sub[sub.method==method].set_index('seed');seeds=baseline.index.intersection(other.index)
            for metric in METRICS:
                diff=(other.loc[seeds,metric]-baseline.loc[seeds,metric]).to_numpy()
                rng=np.random.default_rng(12345);boot=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(axis=1)
                pairs.append({'scenario':scenario,'classifier':clf,'method':method,'metric':metric,'n_pairs':len(diff),'mean_delta':diff.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975)})
    pd.DataFrame(pairs).to_csv(out/'d2_d3_paired_differences.csv',index=False)
    print('Saved to',out)
if __name__=='__main__':main()
