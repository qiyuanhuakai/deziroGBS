"""Exploratory exact-decimal extraction of all author timing records."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
import json,re,hashlib,zipfile,shutil,math
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'.git/iteration68_log_source_pilot_20261110';OUT=ROOT/'.git/iteration68_log_metrics_pilot_20261110'
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(x):
    if isinstance(x,F):return {'rational':str(x)}
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [encode(v) for v in x]
    return x
def save(p,x):assert not p.exists();p.write_bytes((json.dumps(encode(x),indent=2)+'\n').encode())
def stats(values):
    n=len(values);mean=sum(values)/n;variance=sum((x-mean)**2 for x in values)/(n-1)
    return dict(repetitions=n,mean=mean,variance=variance,sem_squared=variance/n,minimum=min(values),maximum=max(values),first=values[0])
def main():
    OUT.mkdir(exist_ok=False);shutil.copy2(Path(__file__),OUT/'source.py');manifest=json.loads((SOURCE/'pilot_manifest.json').read_bytes());assert all(H(SOURCE/n)==h for n,h in manifest['artifacts'].items());rows=[];files=[];unparsed=[]
    with zipfile.ZipFile(SOURCE/'time_log.zip') as z:
        names=[i.filename for i in z.infolist() if not i.is_dir() and not i.filename.startswith('__MACOSX/') and not i.filename.endswith('.DS_Store')]
        for name in names:
            text=z.read(name).decode('utf-8');group=name.split('/')[1];files.append(dict(name=name,sha256=hashlib.sha256(z.read(name)).hexdigest()));buckets=defaultdict(list)
            if group in ('jz2_mode7_PCO','jz2_mode72_PCO'):
                for line in text.splitlines():
                    match=re.fullmatch(r'd=(\d+),D=(\d+),time=([0-9.eE+-]+)',line.strip())
                    if match:d,D,t=match.groups();buckets[int(d),int(D)].append(F(t))
                    elif line and not line.startswith(('average time:','std:')):unparsed.append(dict(name=name,line=line))
                means=re.search(r'average time:Any\[([^]]*)\]',text);stds=re.search(r'std:Any\[([^]]*)\]',text);assert means and stds
                means=[F(x.strip()) for x in means[1].split(',')];stds=[F(x.strip()) for x in stds[1].split(',')];assert len(means)==len(stds)==len(buckets)
                for i,((d,D),values) in enumerate(sorted(buckets.items())):
                    s=stats(values);rows.append(dict(name=name,group=group,d=d,D=D,values=values,statistics=s,logged_mean=means[i],logged_std=stds[i],mean_absolute_error=abs(s['mean']-means[i]),logged_std_squared_to_variance=float(stds[i]**2/s['variance']),logged_std_squared_to_sem_squared=float(stds[i]**2/s['sem_squared'])))
            elif group=='S64_mode2168_PCO':
                dims=re.search(r'_D(\d+)_d(\d+)$',name);assert dims;D,d=map(int,dims.groups());values=[F(m[1]) for m in re.finditer(r'^([0-9.]+) seconds \(',text,re.M)];steps=re.findall(r'Calculating time step (\d+) of (\d+)',text);shapes=re.findall(r'^\((\d+), (\d+), (\d+)\)$',text,re.M);assert len(steps)==len(shapes)==len(values) and set(steps)=={('2167','4336')} and set(shapes)=={(str(D),str(d),str(D))};rows.append(dict(name=name,group=group,d=d,D=D,values=values,statistics=stats(values),logged_step=2167,total_modes=4336))
            elif group=='jz2_mode7_hafnian':
                dims=re.search(r'_d(\d+)_chi(\d+)$',name);assert dims;d,D=map(int,dims.groups());matches=re.findall(r'^Total ([0-9.eE+-]+), a_elem ([0-9.eE+-]+), haf ([0-9.eE+-]+), sigma ([0-9.eE+-]+)\.$',text,re.M);assert matches;values=[F(x[0]) for x in matches];sites=re.findall(r'site:\s+(\d+) j:\s+(\d+)',text);assert len(sites)==d*len(values) and {i for i,j in sites}=={'7'} and {int(j) for i,j in sites}==set(range(d));rows.append(dict(name=name,group=group,d=d,D=D,values=values,components=[dict(total=F(a),a_elem=F(b),haf=F(c),sigma=F(e)) for a,b,c,e in matches],statistics=stats(values)))
            else:raise AssertionError(group)
    assert not unparsed;assert len(files)==66;assert len({(r['group'],r['d'],r['D']) for r in rows})==len(rows)
    paired=[];lookup={(r['group'],r['d'],r['D']):r for r in rows}
    for r in rows:
        if r['group']!='jz2_mode7_PCO':continue
        h=lookup.get(('jz2_mode7_hafnian',r['d'],r['D']))
        if h:paired.append(dict(d=r['d'],D=r['D'],pco_mean=r['statistics']['mean'],hafnian_mean=h['statistics']['mean'],author_cross_hardware_ratio=h['statistics']['mean']/r['statistics']['mean']))
    save(OUT/'rows.json',rows);save(OUT/'paired.json',paired);save(OUT/'summary.json',dict(completed=True,files=files,groups=dict(Counter(r['group'] for r in rows)),panels=len(rows),measurements=sum(len(r['values']) for r in rows),repetition_counts=dict(Counter(str(len(r['values'])) for r in rows)),matched_author_panels=len(paired),unparsed=unparsed,source_sha256=H(Path(__file__)),dependencies={str(SOURCE/'pilot_manifest.json'):H(SOURCE/'pilot_manifest.json'),str(SOURCE/'time_log.zip'):H(SOURCE/'time_log.zip')},scope='Single-parser exploratory author-log metrics,not independent reproduction. All repetitions included; no warm-up exclusion. Ratios are author measurements on different hardware,not local or matched-quality speed claims.'))
    save(OUT/'pilot_manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(dict(logs=len(files),panels=len(rows),measurements=sum(len(r['values']) for r in rows),paired=len(paired),repetitions=dict(Counter(len(r['values']) for r in rows))),indent=2))
if __name__=='__main__':main()
