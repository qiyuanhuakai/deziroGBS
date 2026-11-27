"""Generate the numerical tables and vector figure used by the preprint."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results"
PAPER=ROOT/"papers"/"number-tilted-gbs"


def figure():
    fit=np.load(DATA/"iteration115_tilt_v1/model.npz")
    panel=np.load(DATA/"iteration115_panel_accuracy_v2/pnr_laws.npz")
    robust=json.loads((DATA/"iteration115_robustness_v1/summary.json").read_bytes())
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.85),layout="constrained")
    ax,bx=axes
    n=np.arange(441)
    for key,label,color in [("base","Before correction","#a54c29"),("total","Seven-component mixture","#186882")]:
        err=fit[key]-fit["truth"][:441]
        ax.semilogy(n,np.maximum(abs(err),1e-14),color=color,label=label,lw=1.35)
    ax.set(xlim=(32,220),ylim=(1e-12,1e-2),xlabel="Detected photons, $N$",ylabel="Absolute probability error")
    ax.legend(loc="lower left",frameon=False,fontsize=7)
    x=np.arange(6)
    for field,label,marker,color in [("base_total_tv_upper","Base","o","#a54c29"),
            ("fixed_grid_full_tv_upper","Fixed grid","s","#878888"),("heldout_fit_full_tv_upper","Adaptive grid","^","#186882")]:
        bx.scatter(x,[100*v[field] for v in robust],c=color,marker=marker,s=26,label=label)
    bx.set(yscale="log",xticks=x,xticklabels=["0.35","0.65","1.00","1.20","576","288"],
           ylabel="Total-count TV bound (%)",xlabel="Amplitude multiplier   |   output channels")
    bx.axvline(3.5,color=".75",lw=.7)
    bx.legend(loc="lower left",frameon=False,fontsize=7,ncol=3,bbox_to_anchor=(-.09,-.4))
    for a in axes:a.grid(axis="y",alpha=.15)
    fig.savefig(PAPER/"total-counts.pdf")
    plt.close(fig)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.75),layout="constrained")
    truth=panel["optical_bank"]
    residual=100*(panel["model_bank"]-truth)
    lim=np.max(abs(residual))
    img=ax.imshow(residual[:9,:9].T,cmap="RdBu_r",vmin=-lim,vmax=lim,origin="lower")
    ax.set(xlabel="Photons in bank 34",ylabel="Photons in bank 35",xticks=[0,2,4,6,8],yticks=[0,2,4,6,8])
    fig.colorbar(img,ax=ax,shrink=.8,label="Model minus optical (pp)")
    h=np.load(DATA/"iteration115_pair_banks_v1/counts.npz")
    def distances(a,b):return np.abs(a-b).sum(axis=(1,2))*50
    obs=[h[k]/h[k].sum((1,2),keepdims=True) for k in ("hardware10","hardware11")]
    for v,label,color in [(distances(h['model'],obs[0]),"Model / file 10","#a54c29"),
                  (distances(h['model'],obs[1]),"Model / file 11","#186882"),
                  (distances(*obs),"File 10 / file 11","#666666")]:
        bx.plot(np.sort(v),np.arange(1,len(v)+1)/len(v),label=label,color=color)
    bx.set(xlabel="TV on two-bank click multiplicities (%)",ylabel="Fraction of bank pairs",ylim=(0,1.03))
    bx.legend(frameon=False,loc="lower right",fontsize=7)
    fig.savefig(PAPER/"spatial-errors.pdf")
    plt.close(fig)


def tables():
    s=json.loads((DATA/"iteration115_panel_accuracy_v2/validation.json").read_bytes())
    names=["Direct clicks (GPU)","PNR (GPU)","PE--PNR (CPU)","PE--PNR (GPU)"]
    rows=[]
    for name,r in zip(names,s['measurements']):
        pair=f"{100*r['pair_empirical_tv']:.3f}" if 'pair_empirical_tv' in r else "---"
        rows.append([name,format(r['us'],',.3f'),format(r['mean_us'],',.3f'),
                     f"{100*r['click_empirical_tv']:.3f}",pair])
    for r in s['frugal']:
        rows.append([f"Frugal $K={r['cutoff']}$ (CPU)",f"{r['median_us']:.3f}",f"{r['mean_us']:.3f}",
                     f"{100*r['empirical_tv']:.3f}","---"])
    slash=chr(92)
    (PAPER/"timing-rows.tex").write_text(slash+"newcommand{"+slash+"BenchRows}{%\n"+"".join(" & ".join(r)+" "+slash*2+"%\n" for r in rows)+"}\n",encoding="ascii")
    value=lambda x:f"{100*x:.2f}"
    q=s['null_empirical_tv_quantiles']
    macros={"OpticalNull":q['optical_click'],"TiltNull":q['model_click'],"PairNull":q['optical_pair']}
    text=[]
    for k,(lo,mid,hi) in macros.items():
        text.append(slash+"newcommand{"+slash+k+"}{"+value(mid)+slash+"% ("+value(lo)+"--"+value(hi)+slash+"%)}")
    (PAPER/"null-values.tex").write_text("\n".join(text)+"\n",encoding="ascii")


if __name__=="__main__":
    PAPER.mkdir(exist_ok=True,parents=True)
    figure()
    if (DATA/"iteration115_panel_accuracy_v2/validation.json").exists():tables()
