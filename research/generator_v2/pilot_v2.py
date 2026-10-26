"""Baseline-only sanity pilot. Usage: python pilot_v2.py --seeds 10"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,balanced_accuracy_score
from generators import sample,SCENARIOS

p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--rf-trees',type=int,default=80);args=p.parse_args()
rows=[]
for seed in range(args.seeds):
    for s in SCENARIOS:
        x,y=sample(1000,seed,s,'train');xt,yt=sample(2000,seed,s,'test')
        for name,model in [('LR',make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed))),('RF',RandomForestClassifier(n_estimators=args.rf_trees,random_state=seed,n_jobs=-1))]:
            model.fit(x,y);pr=model.predict_proba(xt)[:,1];pred=(pr>=.5).astype(int)
            rows.append(dict(seed=seed,scenario=s,model=name,train_minority=int(y.sum()),test_minority=int(yt.sum()),ap=average_precision_score(yt,pr),roc_auc=roc_auc_score(yt,pr),f1=f1_score(yt,pred,zero_division=0),balanced_accuracy=balanced_accuracy_score(yt,pred)))
out=Path(__file__).resolve().parent
pd.DataFrame(rows).to_csv(out/'pilot_v2_per_seed.csv',index=False)
sumdf=pd.DataFrame(rows).groupby(['scenario','model'])[['ap','roc_auc','f1','balanced_accuracy']].agg(['mean','std'])
sumdf.to_csv(out/'pilot_v2_summary.csv');print(sumdf.round(4).to_string())
