import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration31_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name)
    rows=json.loads((ROOT/'results/iteration31_panel/rows.json').read_text());small=rows[:16];large=rows[16:];assert len(rows)==20 and all(r['status']=='completed' for r in rows)
    summary=dict(analyzer_sha256=sha,small_maximum_conditional_error=max(r['maximum_conditional_error'] for r in small),small_maximum_proxy_law_tv=max(r['extended_law_tv'] for r in small),small_target_tv_range=[min(r['target_tv'] for r in small),max(r['target_tv'] for r in small)],paired_samples=sum(r['samples'] for r in small),large_successful_samples=sum(r['successful_samples'] for r in large),large_outcomes=[r['sampling_outcome'] for r in large],large_rows=[{k:r[k] for k in ('budget','forest','preparation_seconds','sampling_seconds','successful_samples','seconds')} for r in large])
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');ax[0].plot(np.arange(16),[r['target_tv'] for r in small],'o',label='Forest proxy vs Gaussian target');ax[0].plot(np.arange(16),[r['extended_law_tv'] for r in small],'x',label='Wire native vs independent same-rule table');ax[0].set_xlabel('Fixed small-system case');ax[0].set_ylabel('Total variation');ax[0].set_title('Transport correctness and model error are distinct');ax[0].legend(fontsize=8)
    labels=[f"k={r['budget']}\n{r['forest']}" for r in large];ax[1].bar(labels,[r['sampling_seconds'] for r in large],label='Sampling');ax[1].bar(labels,[r['preparation_seconds'] for r in large],bottom=[r['sampling_seconds'] for r in large],label='Preparation');ax[1].set_ylabel('Seconds');ax[1].set_title('Eight 144-mode strings per configuration');ax[1].legend();fig.suptitle('Iteration 31: explicit scalar transport around the unchanged native forest kernel');fig.savefig(out/'wire_forest_results.png',dpi=140);plt.close(fig)
    summary['unchanged_source']=digest(source)==sha;save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='large_rows'}))
if __name__=='__main__':main()
