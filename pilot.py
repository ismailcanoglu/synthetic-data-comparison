from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'audit_stage1'/'D4_extension'))
from experiment_d1 import sample_dgp
from experiment_d2_d3 import generate
SEEDS=range(10)
rows=[]
for seed in SEEDS:
    for scenario in ['d1','d2','d3']:
        tr_rng=np.random.default_rng(seed)
        te_rng=np.random.default_rng(seed+100000)
        if scenario=='d1':
            Xtr,ytr=sample_dgp(1000,tr_rng);Xte,yte=sample_dgp(2000,te_rng)
        else:
            Xtr,ytr=generate(1000,tr_rng,scenario)
            Xte,yte=generate(2000,te_rng,scenario,shift=(scenario=='d3'))
        for clf_name in ['LR','RF']:
            if clf_name=='LR': clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed))
            else: clf=RandomForestClassifier(n_estimators=80,random_state=seed,n_jobs=-1)
            clf.fit(Xtr,ytr);prob=clf.predict_proba(Xte)[:,1];pred=(prob>=.5).astype(int)
            rows.append(dict(seed=seed,scenario=scenario,model=clf_name,n_train=len(ytr),n_test=len(yte),minority_train=int(ytr.sum()),minority_test=int(yte.sum()),roc_auc=roc_auc_score(yte,prob),average_precision=average_precision_score(yte,prob),f1=f1_score(yte,pred,zero_division=0),balanced_accuracy=balanced_accuracy_score(yte,pred)))
df=pd.DataFrame(rows);df.to_csv(ROOT/'pilot_per_seed.csv',index=False)
summary=df.groupby(['scenario','model'])[['roc_auc','average_precision','f1','balanced_accuracy']].agg(['mean','std']);summary.to_csv(ROOT/'pilot_summary.csv')
print(summary.round(4).to_string())
for metric in ['roc_auc','average_precision','f1','balanced_accuracy']:
    print('\n',metric)
    for model in ['LR','RF']:
        a=df[(df.scenario=='d1')&(df.model==model)].set_index('seed')[metric]
        b=df[(df.scenario=='d3')&(df.model==model)].set_index('seed')[metric]
        d=b-a
        print(model,'D3-D1',round(d.mean(),4),'SD',round(d.std(),4),'positive',int((d>0).sum()),'negative',int((d<0).sum()))
