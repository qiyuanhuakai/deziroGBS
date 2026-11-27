"""Cross-check model masses, the realized GPU transformation, independent peers and held-out hardware."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from iteration100_spatial_v1 import save,manifest
from iteration103_gpu_v1 import I,U,CU,P
from iteration106_field_v1 import DATA
from iteration109_sample_v1 import tv
from iteration110_hardware_v1 import ks

def read(stage):return json.loads((DATA/f'iteration110_{stage}_v1/summary.json').read_text('utf-8'))

def controls():
    lib=ct.CDLL(str(DATA/'iteration110_device_v1/gpu.dll'));fn=lib.transport110;fn.argtypes=[I]*6+[U,CU]+[P]*8
    results=[];host=lambda a:torch.as_tensor(np.asarray(a),device='cuda')
    for pn,wide,bins,target,record,expected_status in [
        (0,0,1,2,[1,0,0,1],0),(1,1,8,7*4+2,[1,0,1,0],0),(1,1,8,0,[1,0,1,0],0),
        (1,0,301,300*4,[0,0,1,0],1),(1,1,301,300*4,[0,0,1,0],0),(1,1,4,0,[0,0,5,0],0)]:
        h=np.array([record]*16,np.int32 if wide else np.uint8);out=host(h)
        size=bins*4;cdf=np.zeros(size,np.uint64);cdf[target:]=2**52
        args=[host(np.array([0,1],np.int32)),host(np.array([0,1,-1,-1],np.int32)),
              host(np.array([0,0,1,0],np.uint32)),host(np.full(size,2**52,np.uint64)),host(cdf),out,host(np.zeros(1,np.int32))]
        assert not fn(len(h),4,pn,wide,2,bins,110690000,0,*[a.data_ptr() for a in args],torch.cuda.current_stream().cuda_stream)
        status=int(args[-1].item());assert status==expected_status
        x=out.cpu().numpy()
        if not status:
            if pn and sum(record)>=bins:assert np.array_equal(h,x)
            else:
                assert np.all((x[:,:2]>0)@np.array([1,2])==target%4)
                if pn:assert np.all(x.sum(1)==target//4)
                else:assert np.array_equal(h[:,2:],x[:,2:])
        results.append(dict(pnr=pn,wide=wide,target=target,status=status))
    return results

def figure(out,hw,profile,peers,collective):
    fig,a=plt.subplots(1,3,figsize=(13.2,3.85),layout='constrained')
    x=np.arange(3);w=.26
    for i,(name,values,color) in enumerate([('Optical',hw['rows'][-1]['before_ks'],'#b38c57'),
        ('Within port',profile['rows'][1]['before_ks'],'#7695b1'),('Spatial profile',profile['rows'][1]['after_ks'],'#236e63')]):
        a[0].bar(x+(i-1)*w,100*np.array(values[:3]),w,label=name,color=color)
    a[0].set_xticks(x,['Clicks','Active ports','Multi-click ports']);a[0].set_ylabel('Held-out hardware KS (pp)');a[0].legend(fontsize=8);a[0].set_title('Jiuzhang 3.0 · PNR readout')
    for field,c,marker in [('gpu_us','#216e64','o')]:
        a[1].bar(x,[v[field] for v in peers['click']],color=[c,'#9380ad','#654f81'])
    a[1].set_yscale('log');a[1].set_ylabel('Complete GPU record (µs)');a[1].set_xticks(x,['Collective','Frugal K2','Frugal K4']);a[1].set_title('16 ports · same GPU and batch')
    rows=collective['rows'][:3]
    for j,(name,color) in enumerate([('Direct clicks','#216e64'),('Photon counts','#a35c40')]):
        a[2].loglog([v['ports'] for v in rows],[v['timing'][j]['us'] for v in rows],marker='o',label=name,color=color)
    a[2].set_xlabel('Physical ports');a[2].set_ylabel('Complete GPU record (µs)');a[2].set_title('Four interfering collective modes per cell');a[2].legend(fontsize=8)
    for ax in a:ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'comparison.png',dpi=145);plt.close(fig)

def run(out):
    out.mkdir(exist_ok=False,parents=True);joint=read('joint');opt=read('device');hw=read('hardware');prof=read('profile')
    independent=read('independent');collect=read('collective');peers=read('peers');bus=read('bus');spectrum=read('spectrum');mixed=read('mixed')
    assert max(mixed['cross_checks'])<1e-9 and len(mixed['physical_ports'])==4 and mixed['source_spectra']==2
    tables=np.load(DATA/'iteration110_joint_v1/tables.npz');oracle=np.load(DATA/'iteration110_independent_v1/laws.npz')
    # Evaluate the published discrete kernel on the independent source distribution.
    checks=[]
    for row in joint['rows']:
        name=row['name'];p=oracle['target'].ravel();q=oracle[name+'_q'].ravel()
        move=tables[name+'_pnr_move'].astype(float)/2**52;cdf=tables[name+'_pnr_cdf']
        corrected=q*(1-move)+(q@move)*np.diff(cdf,prepend=np.uint64(0)).astype(float)/2**52
        error=tv(corrected,p);assert error<1e-9
        checks.append(dict(model=name,real_recurrence_corrected_joint_tv=error,corrected_total_tv=tv(corrected.reshape(-1,256).sum(1),p.reshape(-1,256).sum(1))))
    p=np.load(DATA/'iteration110_profile_v1/histograms.npz');h=p['head_hardware'];ref=h/h.sum()
    prior=json.loads((DATA/'iteration107_peers_v1/summary.json').read_text())
    fr=[]
    for k in (2,4):
        rr=np.load(DATA/f'iteration104_frugal_device_v1/panel_0_frugal32_{k}.npz')['hold']
        z=next(v for v in prior['upstream_gpu'] if v['method']=='frugal32' and v['panel']==0 and v['K']==k)
        fr.append(dict(k=k,us=z['us'],optical_empirical_tv=z['empirical_tv'],hardware_empirical_tv=tv(rr/rr.sum(),ref),shots=int(rr.sum())))
    for pn,z in enumerate(prof['rows']):
        assert p[f'first{pn}'].sum()==z['shots']==262144
        assert np.array_equal(p[f'route{pn}_0_0'],p[f'route{pn}_1_0'])
        assert abs(ks(p[f'route{pn}_1_1'],p['hardware_1'])-z['after_ks'][1])<1e-14
    fo=np.load(DATA/'iteration110_collective_v1/optical.npz');assert tv(fo['full_model'],fo['full_optical'])<1e-10
    pnr_pair=np.load(DATA/'iteration107_projection_v1/pnr.npz')['target']
    rprof=p['pnr_pair'];rne=np.load(DATA/'iteration110_device_v1/histograms.npz')['fitted_1_pair']
    assert rprof.sum()==rne.sum()==262144
    assert len(collect['partial_transpose_minima'])==7 and max(collect['partial_transpose_minima'])<1
    for z in bus['small']:assert z['tv']<=z['total_sector_bound']+1e-10
    assert max(z['half_grid_difference'] for z in bus['wide'])<1e-8
    refs={}
    for stage in ('spectrum','joint','device','hardware','profile','collective','peers','bus','independent','mixed'):
        for path in sorted((DATA/f'iteration110_{stage}_v1').iterdir()):
            refs[path.relative_to(DATA).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    def timings(row):return [v['us'] for v in row['timing']]
    def short(row):return {k:v for k,v in row.items() if k not in ('timing','equal_first8','equal8','eight','timings')}
    optical=[dict(**short(z),full_gpu_us=timings(z),equal_first8_us=z['equal_first8']['us']) for z in opt['rows']]
    deployed=[dict(**short(z),full_gpu_us=timings(z),equal_first8_us=z['equal8']['us']) for z in prof['rows']]
    widths=[dict(ports=z['ports'],cells=z['cells'],fan=z['width'],mean_photons=z['mean_photons'],us=timings(z),
                 batch=z['timing'][0]['batch'],ideal_budget_tv_upper=z['ideal_budget_tv_upper']) for z in collect['rows']]
    save(out,'validation.json',dict(device=opt['device'],jiuzhang_hardware_us=1.27,field_fit=spectrum,
       optical_joint=joint,independent=independent,corrected_cross_check=checks,actual_optical=optical,hardware_profile=deployed,
       hardware_observables=prof['observables'],hardware_train=prof['hardware_train'],hardware_holdout=prof['hardware_holdout'],
       profile_types=prof['unique_profiles'],hardware_split_ks=hw['hardware_split_ks'],frugal_gpu_same_eight_bits=fr,
       pnr_289_861=dict(shots=262144,optical_empirical_tv=tv(rne/rne.sum(),pnr_pair),profile_empirical_tv=tv(rprof/rprof.sum(),pnr_pair),previous_pe=prior['pe_pair']),
       collective=widths,collective_partial_transpose=collect['partial_transpose_minima'],collective_peers=dict(click=[short(z) for z in peers['click']],pnr=peers['pnr']),
       mixed_gaussian_cores=mixed,passive_coupler=bus,cuda_controls=controls(),artifact_sha256=refs))
    figure(out,hw,prof,peers,collect);manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);run(ap.parse_args().output.resolve())
