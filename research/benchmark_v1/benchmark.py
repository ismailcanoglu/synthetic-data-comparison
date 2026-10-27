"""Exploratory controlled benchmark; CTGAN optional, not a preregistered study."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from generators import sample, SCENARIOS, Settings

BASE=Path(__file__).resolve().parent

def seed_for(seed, stream):
    return int(np.random.SeedSequence([seed,stream]).generate_state(1)[0])

def synthesize(method, X, y, n_add, seed, scenario, settings, ctgan_epochs):
    minority=X[y==1]
    t0=time.perf_counter()
    if method=='smote':
        sm=SMOTE(sampling_strategy={1:int(np.sum(y==1)+n_add)},k_neighbors=5,random_state=seed_for(seed,41))
        xa,ya=sm.fit_resample(X,y)
        gen=xa[len(X):]
    elif method=='oracle':
        # Training DGP only. For shifted scenarios this intentionally does not see test intervention.
        gen=[]; attempts=0
        while sum(len(a) for a in gen)<n_add and attempts<100:
            xx,yy=sample(max(300,n_add*12),seed_for(seed,100+attempts),scenario,'train',settings)
            gen.append(xx[yy==1]);attempts+=1
        gen=np.vstack(gen)[:n_add]
    elif method=='ctgan':
        try:
            from ctgan import CTGAN
        except ImportError as e:
            raise RuntimeError('CTGAN unavailable; install optional ctgan package and pin its version') from e
        # CTGAN is fitted on training fold only. Fit joint X,y and condition on y=1.
        import inspect
        df=pd.DataFrame(X,columns=[f'x{i}' for i in range(X.shape[1])]);df['target']=y
        kw={'epochs':ctgan_epochs,'verbose':False}
        sig=inspect.signature(CTGAN)
        if 'enable_gpu' in sig.parameters:kw['enable_gpu']=False
        elif 'cuda' in sig.parameters:kw['cuda']=False
        np.random.seed(seed_for(seed,51))  # CTGAN may use global RNG internally
        model=CTGAN(**kw);model.fit(df,discrete_columns=['target'])
        gen=[]
        for attempt in range(40):
            try: new=model.sample(max(n_add*2,100),condition_column='target',condition_value=1)
            except Exception: new=model.sample(max(n_add*3,150))
            subset=new.loc[new.target.round().astype(int)==1, [f'x{i}' for i in range(X.shape[1])]]
            if len(subset):gen.append(subset.to_numpy(dtype=float))
            if sum(len(z) for z in gen)>=n_add:break
        if not gen or sum(len(z) for z in gen)<n_add:raise RuntimeError('CTGAN failed to generate enough minority rows')
        gen=np.vstack(gen)[:n_add]
    else:raise ValueError(method)
    if not np.isfinite(gen).all():raise ValueError('non-finite generated samples')
    return gen,time.perf_counter()-t0

def quality(generated,reference,train_minority,scaler):
    # Reference and train are class-1 ONLY. Scaler fitted on real train fold ONLY.
    a=scaler.transform(generated); b=scaler.transform(reference); t=scaler.transform(train_minority)
    # RBF bandwidth fixed using reference-only distances; no synthetic samples in tuning.
    from scipy.spatial.distance import pdist
    dist=pdist(b,'sqeuclidean'); med=np.median(dist[dist>0]) if np.any(dist>0) else 1.0
    gamma=1/max(med,1e-10)
    mmd=float(rbf_kernel(a,a,gamma=gamma).mean()+rbf_kernel(b,b,gamma=gamma).mean()-2*rbf_kernel(a,b,gamma=gamma).mean())
    nn=NearestNeighbors(n_neighbors=1).fit(t)
    d=nn.kneighbors(a,return_distance=True)[0][:,0]
    # Standardized distances are not comparable across differently scaled datasets without protocol.
    corr_g=np.corrcoef(a,rowvar=False);corr_r=np.corrcoef(b,rowvar=False)
    return {'mmd2_biased':max(0,mmd),'corr_mae':float(np.mean(np.abs(corr_g[np.triu_indices(a.shape[1],1)]-corr_r[np.triu_indices(a.shape[1],1)]))),
            'nearest_train_distance':float(np.mean(d)), 'unique_rounded_4dp':float(np.unique(np.round(generated,4),axis=0).shape[0]/len(generated)),
            'structural_residual':float(np.mean(np.abs(generated[:,3]-generated[:,0]*generated[:,1])+np.abs(generated[:,4]-generated[:,0]**2)+np.abs(generated[:,5]-np.sin(generated[:,2])))/3)}

def run(seeds,methods,scenarios,trees,epochs,output,fit_n=750,val_n=250,test_n=2000):
    output.mkdir(parents=True,exist_ok=True)
    records=[]; qr=[];failures=[]
    for scenario in scenarios:
      for seed in seeds:
        Xall,yall=sample(fit_n+val_n,seed,scenario,'train')
        Xfit,Xval,yfit,yval=train_test_split(Xall,yall,test_size=val_n,random_state=seed_for(seed,10),stratify=yall)
        Xtest,ytest=sample(test_n,seed,scenario,'test')
        # independent reference from unshifted train DGP; minority-only reference drawn without fitting
        refx,refy=sample(10000,seed_for(seed,71),scenario,'train')
        reference=refx[refy==1][:500]
        scaler=StandardScaler().fit(Xfit)
        n_add=int(sum(yfit==0)-sum(yfit==1))
        if sum(yfit==1)<=5:raise ValueError('Not enough minority observations for SMOTE k=5')
        for method in methods:
          try:
            if method in ('baseline','class_weight'):
                xa,ya=Xfit,yfit;elapsed=0;gen=None
            else:
                gen,elapsed=synthesize(method,Xfit,yfit,n_add,seed,scenario,Settings(),epochs)
                xa=np.vstack([Xfit,gen]);ya=np.concatenate([yfit,np.ones(len(gen),dtype=int)])
                q=quality(gen,reference,Xfit[yfit==1],scaler)
                qr.append({'scenario':scenario,'seed':seed,'method':method,'n_added':len(gen),'generation_seconds':elapsed,**q})
            for clf in ('LR','RF'):
                weighted=(method=='class_weight')
                model=(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced' if weighted else None,random_state=seed_for(seed,82))) if clf=='LR' else RandomForestClassifier(n_estimators=trees,n_jobs=-1,class_weight='balanced' if weighted else None,random_state=seed_for(seed,83)))
                model.fit(xa,ya)
                pval=model.predict_proba(Xval)[:,1];ptest=model.predict_proba(Xtest)[:,1]
                # Val-only F1 threshold selection; tie broken by first/smallest threshold.
                grid=np.linspace(.05,.95,91)
                threshold=float(grid[np.argmax([f1_score(yval,pval>=t,zero_division=0) for t in grid])])
                for policy,t in [('default',.5),('val_tuned',threshold)]:
                    pred=(ptest>=t)
                    records.append({'scenario':scenario,'seed':seed,'method':method,'model':clf,'threshold_policy':policy,'threshold':t,'n_fit':len(Xfit),'n_val':len(Xval),'n_test':len(Xtest),'n_added':0 if gen is None else len(gen),
                        'average_precision':average_precision_score(ytest,ptest),'roc_auc':roc_auc_score(ytest,ptest),'minority_f1':f1_score(ytest,pred,zero_division=0),'balanced_accuracy':balanced_accuracy_score(ytest,pred)})
          except Exception as e:
            failures.append({'scenario':scenario,'seed':seed,'method':method,'error':repr(e)})
    pd.DataFrame(records).to_csv(output/'results_per_seed.csv',index=False)
    if records:
        agg=pd.DataFrame(records).groupby(['scenario','method','model','threshold_policy'],as_index=False).agg(n=('seed','size'),ap_mean=('average_precision','mean'),ap_sd=('average_precision','std'),f1_mean=('minority_f1','mean'),f1_sd=('minority_f1','std'),roc_auc_mean=('roc_auc','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'))
        agg.to_csv(output/'results_summary.csv',index=False)
    pd.DataFrame(qr).to_csv(output/'quality_per_seed.csv',index=False)
    pd.DataFrame(failures,columns=['scenario','seed','method','error']).to_csv(output/'failures.csv',index=False)
    manifest={'status':'EXPLORATORY_NOT_CONFIRMATORY','seeds':list(seeds),'scenarios':scenarios,'methods':methods,'rf_trees':trees,'ctgan_epochs':epochs,'fit_n':fit_n,'validation_n':val_n,'test_n':test_n,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return records,qr,failures

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--rf-trees',type=int,default=80);ap.add_argument('--ctgan-epochs',type=int,default=100);ap.add_argument('--methods',nargs='+',default=['baseline','class_weight','smote','oracle']);ap.add_argument('--scenarios',nargs='+',default=list(SCENARIOS));ap.add_argument('--output',default='outputs');a=ap.parse_args()
    run(range(a.seeds),a.methods,a.scenarios,a.rf_trees,a.ctgan_epochs,Path(a.output))
