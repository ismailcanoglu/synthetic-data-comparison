"""D4: threshold vs class-weight vs augmentation; held-out validation, untouched test.
Usage: python experiment_d4.py --skip-ctgan --seeds 11 23
       python experiment_d4.py --epochs 100
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from experiment_d1 import sample_dgp, ctgan_minority
from experiment_d2_d3 import generate

SEEDS=[11,23,37,51,79]
ROOT=Path(__file__).resolve().parent

def classifier(name,seed,weighted):
    weight='balanced' if weighted else None
    if name=='logistic_regression':
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed,class_weight=weight))
    return RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=-1,class_weight=weight)

def choose_threshold(y,prob):
    # Deterministic, preregistered grid; validation data only.
    grid=np.linspace(.05,.95,91)
    scores=[f1_score(y,prob>=t,zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])

def data(seed,scenario):
    if scenario=='d1':
        tr=sample_dgp(1000,np.random.default_rng(seed));te=sample_dgp(2000,np.random.default_rng(seed+100000))
    else:
        tr=generate(1000,np.random.default_rng(seed),scenario)
        te=generate(2000,np.random.default_rng(seed+100000),scenario,shift=(scenario=='d3'))
    return (*tr,*te)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--skip-ctgan',action='store_true')
    ap.add_argument('--scenarios',nargs='+',choices=['d1','d2','d3'],default=['d1','d2','d3'])
    args=ap.parse_args()
    out=ROOT/'results_d4';out.mkdir(exist_ok=True)
    rows=[];failures=[]
    for scenario in args.scenarios:
        for seed in args.seeds:
            X,y,Xtest,ytest=data(seed,scenario)
            Xfit,Xval,yfit,yval=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
            n_add=int((yfit==0).sum()-(yfit==1).sum())
            Xsm,ysm=SMOTE(random_state=seed,k_neighbors=5).fit_resample(Xfit,yfit)
            if scenario=='d1':
                Xor,_=sample_dgp(n_add,np.random.default_rng(seed+200000),minority_only=True)
            else:
                Xor,_=generate(n_add,np.random.default_rng(seed+200000),scenario,minority_only=True)
            methods={'baseline':(Xfit,yfit,False),'class_weight_balanced':(Xfit,yfit,True),
                     'smote':(Xsm,ysm,False),
                     'oracle_simulation':(np.vstack([Xfit,Xor]),np.r_[yfit,np.ones(n_add,dtype=int)],False)}
            if not args.skip_ctgan:
                try:
                    Xct,_=ctgan_minority(Xfit,yfit,n_add,seed,args.epochs)
                    methods['ctgan']=(np.vstack([Xfit,Xct]),np.r_[yfit,np.ones(n_add,dtype=int)],False)
                except Exception as exc:
                    failures.append(dict(scenario=scenario,seed=seed,method='ctgan',error=str(exc)))
            for method,(Xt,yt,weighted) in methods.items():
                for name in ('logistic_regression','random_forest'):
                    model=classifier(name,seed,weighted).fit(Xt,yt)
                    valprob=model.predict_proba(Xval)[:,1]
                    testprob=model.predict_proba(Xtest)[:,1]
                    tuned=choose_threshold(yval,valprob)
                    for mode,t in [('default',.5),('validation_tuned',tuned)]:
                        pred=(testprob>=t).astype(int)
                        rows.append(dict(scenario=scenario,seed=seed,method=method,classifier=name,
                            threshold_mode=mode,threshold=t,n_fit=len(Xfit),n_validation=len(Xval),n_added=len(Xt)-len(Xfit),
                            f1_minority=f1_score(ytest,pred,zero_division=0),
                            f1_macro=f1_score(ytest,pred,average='macro',zero_division=0),
                            precision=precision_score(ytest,pred,zero_division=0),
                            recall=recall_score(ytest,pred,zero_division=0),
                            balanced_accuracy=balanced_accuracy_score(ytest,pred),
                            pr_auc=average_precision_score(ytest,testprob),auroc=roc_auc_score(ytest,testprob)))
            pd.DataFrame(rows).to_csv(out/'d4_per_seed.csv',index=False)
            pd.DataFrame(failures,columns=['scenario','seed','method','error']).to_csv(out/'d4_failures.csv',index=False)
            print(f'Completed {scenario} seed {seed}',flush=True)
    df=pd.DataFrame(rows)
    metrics=['f1_minority','f1_macro','precision','recall','balanced_accuracy','pr_auc','auroc','threshold']
    df.groupby(['scenario','classifier','method','threshold_mode'])[metrics].agg(['mean','std']).to_csv(out/'d4_summary.csv')
    # Paired comparison with default baseline under the same train/validation split.
    baseline=df[(df.method=='baseline')&(df.threshold_mode=='default')].set_index(['scenario','seed','classifier'])
    diffs=[]
    for (scenario,clf,method,mode),sub in df.groupby(['scenario','classifier','method','threshold_mode']):
        sub=sub.set_index(['scenario','seed','classifier'])
        common=sub.index.intersection(baseline.index)
        for metric in ('f1_minority','pr_auc','balanced_accuracy'):
            diff=(sub.loc[common,metric]-baseline.loc[common,metric]).to_numpy()
            diffs.append(dict(scenario=scenario,classifier=clf,method=method,threshold_mode=mode,
                              metric=metric,n_pairs=len(diff),mean_delta=float(diff.mean()),sd_delta=float(diff.std(ddof=1)) if len(diff)>1 else np.nan))
    pd.DataFrame(diffs).to_csv(out/'d4_paired_differences.csv',index=False)
    print('Saved:',out)

if __name__=='__main__':main()
