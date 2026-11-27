"""Verify distributions, raw transfer captures, timing rows and the round-112 archive manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from iteration111_sampling_v1 import DATA,ROOT


def run(out,captures):
    out.mkdir(parents=True,exist_ok=True)
    def d(name):return DATA/f'iteration112_{name}_v1'
    def read(name,file):return json.loads((d(name)/(file+'.json')).read_text())
    def load(name,file):return np.load(d(name)/(file+'.npz'))
    def tv(p,q):return float(abs(p-q).sum()/2)
    def ks(p,q):return float(abs((p-q).cumsum()).max())
    def timing(t):return {k:float(np.median(v)) for k,v in t.items()}
    small=[];curves=read('fit','fits');refs=load('cases','laws')
    fit=load('fit','fits');coarse=read('coarse','coarse');sam=load('coarse','records')
    for row in curves:
        b,n=row['buses'],row['budget'];p=fit[f'b{b}n{n}_law'];true=refs[f'truth{b}']
        assert abs(p.sum()-1)<1e-10 and p.min()>-1e-12
        np.testing.assert_allclose(tv(p,true),row['click_tv'],atol=1e-13)
    for j,rec in enumerate(coarse):
        b=rec['buses'];pair=refs[f'pair{b}'];ss=rec['shots']
        label='two_peers' if b==2 else 'four_peers'
        peers=read(label,'peers');z=load(label,'peers')
        row={k:v for k,v in rec.items() if k!='timing'};row['us']=timing(rec['timing'])
        assert sam[f'{j}_hist'].sum()==ss
        np.testing.assert_allclose(tv(sam[f'{j}_hist']/ss,z['truth']),rec['empirical_click_tv'])
        if rec['radix']==2:
            row['peers']=[]
            for f in peers['frugal']:
                k=f['k'];backend=f['backend']
                np.testing.assert_allclose(tv(z[f'hist{k}{backend}']/f['shots'],z['truth']),f['empirical_tv'])
                if backend=='cpu':np.testing.assert_allclose(tv(z[f'frugal{k}'],z['truth']),f['model_tv'])
                row['peers'].append({k:v for k,v in f.items() if k!='times'})
        else:
            np.testing.assert_allclose(tv(sam[f'{j}_model_pair'],pair),rec['pair_model_tv'])
            np.testing.assert_allclose(ks(sam[f'{j}_model_total'],z['total']),rec['total_model_ks'])
            row['peers']=[]
            for f in peers['pe']:
                key=f['backend'];assert ss==f['shots']
                np.testing.assert_allclose(tv(z[key+'_pair']/ss,pair),f['pair_empirical_tv'])
                np.testing.assert_allclose(ks(z[key+'_total']/ss,z['total']),f['total_empirical_ks'])
                row['peers'].append({k:v for k,v in f.items() if k!='times'})
        small.append(row)
    rows=read('emit','emit');arr=load('emit','records');wide=[]
    for r in rows:
        name=r['name'];n=r['ports'];raw=arr[name+'_first']
        first=raw if r['pnr'] else np.unpackbits(raw,axis=1,count=n,bitorder='little')
        assert np.array_equal(first.sum(1),arr[name+'_totals'][:len(first)]) and len(first)==4
        if r['pnr']:assert first.max()>1
        wide.append({**{k:v for k,v in r.items() if k!='timing'},'us':timing(r['timing'])})
    fock=[]
    z=load('records','records')
    for r in read('records','records'):
        n=r['ports'];s=f'b{r["buses"]}p{int(r["pnr"])}c{r["cells"]}'
        a=z[s+'_first'] if r['pnr'] else np.unpackbits(z[s+'_first'],axis=1,count=n,bitorder='little')
        assert np.array_equal(a.sum(1),z[s+'_total'][:len(a)])
        b=np.sqrt(r['collision_probability_upper'])+r['selected_tail']+r['register_tail']+r['rounding']*r['cells']+r['cap']*r['effect_correction']
        np.testing.assert_allclose(b,r['tv_budget'])
        fock.append({**{k:v for k,v in r.items() if k!='timing'},'us':timing(r['timing'])})
    bounds=read('marginals','bounds')+read('count_marginals','bounds')
    for r in bounds:
        np.testing.assert_allclose(r['full_tv_budget'],min(1,r['projection']+.5*np.hypot(r['first_order_l2'],r['higher_order_l2_upper'])+r['register']))
    hw=read('hardware','hardware');files=read('hardware','sources');z=load('hardware','observed');fused=read('fused','fused')
    triangle=np.triu_indices(144,1)
    def pairgap(p,q):
        diff=p-q;s=diff.diagonal()
        a=.5*(abs(diff)+abs(s[:,None]-diff)+abs(s[None,:]-diff)+abs(diff-s[:,None]-s[None,:]))
        return [float(f(a[triangle])) for f in (np.mean,np.max)]
    transfer=[];blocks={}
    for f in files:
        path=captures/f['member'];assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256']
        data=np.memmap(path,np.uint8,mode='r').reshape(-1,144);assert len(data)==f['rows']
        edges=np.linspace(0,len(data),33,dtype=int)
        joints=[]
        for lo,hi in zip(edges,edges[1:]):
            a=(data[lo:hi]!=0).astype(np.float32);joints.append((a.T@a).astype(np.int64))
        file=f['file']
        np.testing.assert_array_equal(np.sum(joints,axis=0),z[f'hardware{file}_joint'])
        for route,row in enumerate(hw['rows']):
            before=z[f'before{route}_joint']/row['shots'];after=z[f'after{route}_joint']/row['shots']
            v=row['blind'][str(file)]
            for kind,p in [('before',before),('after',after)]:
                np.testing.assert_allclose(pairgap(p,z[f'hardware{file}_joint']/len(data)),[v[kind]['pair_mean_tv'],v[kind]['pair_max_tv']])
            arr=np.array([[pairgap(p,h/(hi-lo)) for p in (before,after)] for h,lo,hi in zip(joints,edges,edges[1:])])
            delta=arr[:,0,0]-arr[:,1,0];blocks[f'file{file}_route{route}']=arr
            transfer.append(dict(file=file,pnr=bool(route),test=len(data),before=v['before'],after=v['after'],
                contiguous_blocks=len(delta),blocks_improved=int(np.sum(delta>0)),smallest_mean_tv_reduction=float(delta.min()),
                largest_mean_tv_reduction=float(delta.max()),brightness_pilot=row['adapted'][str(file)],
                fused_us=float(np.median(fused[route]['times']['16384'][2]))))
    np.savez_compressed(out/'time_windows.npz',**blocks)
    panels=read('panels','panels')
    pa=load('panels','panels')
    for r in panels['results']:
        k=r['kind'];name=r['method']
        key=name if name.startswith('frugal') else name+'_'+k
        if name.startswith('hardware'):continue
        a=pa[key]/r['shots_per_panel']
        for ref in ('optical','file10','file11'):
            q=pa[k+'_optical'] if ref=='optical' else pa[ref+'_'+k]
            v=abs(a-q/q.sum(1,keepdims=True)).sum(1)/2
            np.testing.assert_allclose(v,r[ref]['per_panel'],atol=1e-13)
    result=dict(threads_cpu=8,gpu='NVIDIA GeForce RTX 5080 Laptop GPU',jiuzhang_reference_us=1.27,
        small=small,matched_cache=[{k:v for k,v in row.items() if k!='times'} for row in read('cache','cache')],wide_marginals=wide,wide_marked=fock,unequal=read('unequal','unequal'),
        marginal_bounds=bounds,transfer=transfer,real_capture_difference=hw['cross_capture'],
        additional_panels=[{**{k:v for k,v in p.items() if not isinstance(v,dict)},**{k:v['mean'] for k,v in p.items() if isinstance(v,dict)}} for p in panels['results']],
        panel_frugal_timing=panels['frugal_timing'],independent=read('independent','checks'),sources=files)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    fig,ax=plt.subplots(1,3,figsize=(12,3.65),layout='constrained')
    for b in (2,3,4):
        seq=[r for r in curves if r['buses']==b]
        ax[0].semilogy([p['branches'] for p in seq],[100*p['click_tv'] for p in seq],'o-',label=f'{b} phases')
    ax[0].set(xlabel='Positive coherent branches',ylabel='Complete click TV (%)',title='13 heterogeneous ports');ax[0].legend(frameon=False)
    for m in (8,10,12):
        seq=[r for r in bounds if r['readout']=='click' and r['nodes']==m]
        ax[1].loglog([r['ports'] for r in seq],[100*r['full_tv_budget'] for r in seq],'o-',label=f'{m} nodes/axis')
    ax[1].set(xlabel='Physical ports',ylabel='Full TV budget (%)',title='One shared phase');ax[1].legend(frameon=False)
    names=['10 click','10 PNR','11 click','11 PNR'];x=np.arange(4)
    for i,(side,color) in enumerate([('before','#adbcc4'),('after','#26756f')]):
        ax[2].bar(x+(i-.5)*.32,[100*r[side]['pair_mean_tv'] for r in transfer],.32,color=color,label=side)
    ax[2].set_xticks(x,names,rotation=20);ax[2].set(ylabel='Mean spatial pair TV (%)',title='Independent Jiuzhang captures')
    ax[2].legend(frameon=False);fig.savefig(out/'comparison.png',dpi=180)
    paths=list(DATA.glob('iteration112*/*'))+list((ROOT/'experiments').glob('*iteration112*.py'))+list((ROOT/'native').glob('*112*'))
    paths.extend([DATA/'iteration111_heterogeneous_v1/laws.npz',DATA/'iteration110_collective_v1/optical.npz',
                  DATA/'iteration111_transport_v1/holdout.npz'])
    digests={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file() and p!=out/'manifest.json'}
    (out/'manifest.json').write_text(json.dumps(digests,indent=2)+'\n',encoding='utf8')
    print('verified',len(digests),'files and 128 chronological comparisons',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--captures',type=Path,default=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent')
    args=ap.parse_args()
    with threadpool_limits(limits=3):run(args.output.resolve(),args.captures.resolve())
