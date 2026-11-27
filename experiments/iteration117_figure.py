"""Regenerate the cross-power bank comparison and the resolved PNR bounds."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from iteration117_quartic import OUT
from iteration115_visible_v1 import ROOT

fig,axes=plt.subplots(1,2,figsize=(7.2,2.55),layout="constrained")
left,right=axes
colors=["#007483","#ad5428","#534987"]
for power,color in zip(("low","median","high"),colors):
    data=np.array(json.loads((OUT/("banks_"+power+".json")).read_text())["tv_by_bank"])*100
    for i,style in enumerate(("--","-")):
        left.step(np.sort(data[:,i]),np.arange(1,145)/144,where="post",color=color,ls=style,lw=1.3)
left.set(xlabel="Complete single-bank click TV (%)",ylabel="Fraction of 144 banks",ylim=(0,1.04),xlim=(0,.55))
left.legend([Line2D([],[],color=c) for c in colors],["Low","Median","High"],fontsize=7,frameon=False,loc="lower right")
left.set_title("A   Three optical calibrations",loc="left",fontsize=10)
for prefix,color,label in [("","#007483","Four-photon core"),("transport_","#ad5428","Coherence restored")]:
    d=json.loads((OUT/(prefix+"resolved_pnr.json")).read_text())
    n=np.array([v["max_count"] for v in d]);lo=np.array([v["pnr_lower"] for v in d])*100
    hi=np.array([v["pnr_upper"] for v in d])*100
    right.fill_between(n,lo,hi,color=color,alpha=.11)
    right.plot(n,hi,"o-",color=color,ms=3,lw=1.2,label=label);right.plot(n,lo,":",color=color,lw=1.1)
right.set(xlabel="Largest panel count resolved exactly",ylabel="Full 16-port PNR TV (%)",xticks=range(7),ylim=(.34,2.3))
right.legend(fontsize=7,frameon=False)
right.set_title("B   Stress-panel PNR bounds",loc="left",fontsize=10)
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
dest=ROOT/"papers/number-tilted-gbs/quartic-readout.pdf"
fig.savefig(dest,metadata={"Title":"Spatial coherence and resolved photon-number bounds","Author":"qiyuanhuakai"})
