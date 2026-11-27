"""Create the full-record bound and complete panel accuracy figure for the paper."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iteration118_auxiliary import OUT
from iteration115_visible_v1 import DATA,ROOT


def read(folder,name):
    return json.loads((folder/(name+".json")).read_text(encoding="utf8"))


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                          "axes.spines.top":False,"axes.spines.right":False})
    fig,(a,b)=plt.subplots(1,2,figsize=(7.25,2.7),gridspec_kw={"width_ratios":[.85,1.22]})
    slate="#395a76";teal="#137268"
    caps=["4 / 4","10 / 10","14 / 10"]
    vals=[52.049,100*read(OUT,"cap10_128")["full_TV_upper"],100*read(OUT,"cap14_weak10_2")["full_TV_upper"]]
    a.bar(caps,vals,color=[slate,slate,teal],width=.54)
    for i,y in enumerate(vals):a.text(i,y+1.1,f"{y:.2f}%",ha="center",fontsize=8)
    a.set_ylim(0,61);a.set_ylabel("Full-record TV upper bound (%)")
    a.set_xlabel("Core photon caps by spectrum")
    a.set_title("1,152 detector ports",loc="left",fontweight="bold")
    for i,(name,tag) in enumerate([("34, 35",""),("71, 72","_71_72"),("102, 143","_102_143")]):
        row=read(OUT,"panel_7_5_28"+tag);dist=read(OUT,"panel_distance"+tag)
        lo,hi=100*row["bank_pnr_modulo_tv"],100*dist["panel_trace_upper"]
        b.plot([lo,hi],[2-i+.09]*2,color=slate,linewidth=2,label="Full PNR bracket" if i==0 else None)
        b.scatter([lo,hi],[2-i+.09]*2,marker="|",s=60,color=slate)
        b.scatter([100*row["full_panel_click_tv"]],[2-i-.09],color=teal,s=28,
                  label="Full click TV" if i==0 else None)
    b.set_xscale("log");b.set_xlim(.0017,.23);b.set_ylim(-.45,2.5)
    b.set_yticks(range(3),["102, 143","71, 72","34, 35"])
    b.set_xlabel("Total variation distance (%)");b.set_title("Complete 16-port distributions",loc="left",fontweight="bold")
    b.grid(axis="x",alpha=.2);b.legend(loc="upper center",bbox_to_anchor=(.48,-.25),ncol=2,frameon=False)
    fig.tight_layout(w_pad=2.1)
    fig.savefig(ROOT/"papers/number-tilted-gbs/full-record-bounds.pdf",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
