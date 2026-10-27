from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t
p=Path(__file__).resolve().parent/'outputs'
a=pd.read_csv(p/'results_per_seed.csv')
rows=[]
for (scenario,model,policy),g in a.groupby(['scenario','model','threshold_policy']):
    base=g[g.method=='baseline'].set_index('seed')
    for method in sorted(set(g.method)-{'baseline'}):
        other=g[g.method==method].set_index('seed')
        for metric in ['average_precision','minority_f1','balanced_accuracy']:
            d=(other[metric]-base[metric]).dropna().to_numpy()
            n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1)) if n>1 else float('nan')
            half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
            rows.append({'scenario':scenario,'model':model,'threshold_policy':policy,'contrast':f'{method}-baseline','metric':metric,'n_pairs':n,'mean_difference':mean,'sd_difference':sd,'exploratory_t_interval_low':mean-half,'exploratory_t_interval_high':mean+half,'note':'pilot only; CI assumes seed-level differences approximately iid; not confirmatory'})
pd.DataFrame(rows).to_csv(p/'paired_effects_exploratory.csv',index=False)
q=pd.read_csv(p/'quality_per_seed.csv')
q.groupby(['scenario','method'],as_index=False)[['mmd2_biased','corr_mae','nearest_train_distance','structural_residual','generation_seconds']].mean().to_csv(p/'quality_summary.csv',index=False)
print('paired effects:',len(rows),'quality rows:',len(q))
