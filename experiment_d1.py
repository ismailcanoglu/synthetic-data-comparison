"""Conference D1: run with python experiment_d1.py [--seeds 11 23 37 51 79] [--epochs 100] [--skip-ctgan]."""
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent

def sample_dgp(n, rng, minority_only=False):
    y = np.ones(n, dtype=int) if minority_only else rng.binomial(1, .10, size=n)
    z = rng.normal(size=(n, 3)); z[:, 0] += 1.1*y; z[:, 1] -= .8*y
    x = np.column_stack([z, z[:,0]*z[:,1], z[:,0]**2, np.sin(z[:,2])])
    x += rng.normal(0, .08, size=x.shape)
    return x, y

def ctgan_minority(X, y, n, seed, epochs):
    from ctgan import CTGAN
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    columns = [f'f{i}' for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns); frame['target'] = y.astype(int)
    model = CTGAN(epochs=epochs, batch_size=100, verbose=False, cuda=False)
    model.fit(frame, discrete_columns=['target'])
    parts=[]; count=0
    for _ in range(30):
        batch = model.sample(max(200, (n-count)*2), condition_column='target', condition_value=1)
        batch['target'] = pd.to_numeric(batch['target'], errors='coerce')
        batch = batch.loc[batch['target'].eq(1)].copy()
        for col in columns: batch[col] = pd.to_numeric(batch[col], errors='coerce')
        batch = batch.dropna(subset=columns)
        if len(batch): parts.append(batch[columns].to_numpy(float)); count += len(batch)
        if count >= n: break
    if count < n: raise RuntimeError(f'CTGAN produced only {count}/{n} valid minority samples')
    return np.vstack(parts)[:n], np.ones(n, dtype=int)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[11,23,37,51,79]); ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--skip-ctgan', action='store_true'); args=ap.parse_args()
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    rows=[]; failures=[]
    for seed in args.seeds:
        Xtr,ytr=sample_dgp(1000,np.random.default_rng(seed))
        Xte,yte=sample_dgp(2000,np.random.default_rng(seed+100000))
        n_add=int(sum(ytr==0)-sum(ytr==1))
        Xsm,ysm=SMOTE(random_state=seed,k_neighbors=5).fit_resample(Xtr,ytr)
        Xor,yor=sample_dgp(n_add,np.random.default_rng(seed+200000),minority_only=True)
        methods={'baseline':(Xtr,ytr), 'smote':(Xsm,ysm), 'oracle_simulation':(np.vstack((Xtr,Xor)),np.r_[ytr,yor])}
        if not args.skip_ctgan:
            try:
                Xct,yct=ctgan_minority(Xtr,ytr,n_add,seed,args.epochs)
                methods['ctgan']=(np.vstack((Xtr,Xct)),np.r_[ytr,yct])
            except Exception as e:
                failures.append({'seed':seed,'method':'ctgan','error':str(e)})
                print(f'CTGAN failed for seed {seed}: {e}')
        for method,(X,y) in methods.items():
            classifiers={'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed)), 'random_forest':RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=-1)}
            for name,clf in classifiers.items():
                clf.fit(X,y); pred=clf.predict(Xte); prob=clf.predict_proba(Xte)[:,1]
                rows.append({'seed':seed,'method':method,'classifier':name,'n_train':len(X),'n_added':len(X)-len(Xtr), 'f1_macro':f1_score(yte,pred,average='macro'), 'f1_minority':f1_score(yte,pred,pos_label=1,zero_division=0), 'balanced_accuracy':balanced_accuracy_score(yte,pred), 'auroc':roc_auc_score(yte,prob), 'pr_auc':average_precision_score(yte,prob)})
        pd.DataFrame(rows).to_csv(out/'d1_per_seed.csv',index=False)
        pd.DataFrame(failures,columns=['seed','method','error']).to_csv(out/'failures.csv',index=False)
        print('Completed seed',seed)
    df=pd.DataFrame(rows)
    metrics=['f1_macro','f1_minority','balanced_accuracy','auroc','pr_auc']
    df.groupby(['method','classifier'])[metrics].agg(['mean','std']).to_csv(out/'d1_summary.csv')
    pairs=[]
    for classifier in df.classifier.unique():
        base=df[(df.method=='baseline')&(df.classifier==classifier)].set_index('seed')
        for method in df.method.unique():
            if method=='baseline':continue
            comp=df[(df.method==method)&(df.classifier==classifier)].set_index('seed')
            common=base.index.intersection(comp.index)
            for metric in metrics:
                delta=(comp.loc[common,metric]-base.loc[common,metric]).to_numpy()
                if len(delta)==0:continue
                # Bootstrap resampling paired seed-level differences (descriptive CI, not independent test sets)
                rng=np.random.default_rng(12345); draws=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
                pairs.append({'classifier':classifier,'method':method,'metric':metric,'n_pairs':len(delta),'mean_delta':delta.mean(),'ci95_low':np.quantile(draws,.025),'ci95_high':np.quantile(draws,.975)})
    pd.DataFrame(pairs).to_csv(out/'paired_differences.csv',index=False)
    print('\n',df.groupby(['method','classifier'])[metrics].mean().round(3))
    print('\nSaved results to:',out)
if __name__=='__main__':main()
