"""Freeze an optical-versus-capture event on file 10; evaluate files 12 and 13."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_projection_review_v3 import factor
from iteration115_visible_v1 import DATA,ROOT,hardware
OUT=DATA/"iteration119_weights"

CACHE=ROOT.parent/"external_sources"/"jiuzhang3_iteration119_holdout"


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def void(factors,banks):
    if not len(banks):return 1.
    sites=(8*np.array(banks)[:,None]+np.arange(8)).ravel()
    axes=np.r_[sites,1152+sites];value=1.
    for u,e in factors:
        mat=u[axes];sign,log=np.linalg.slogdet(np.eye(len(axes))+(mat*e)@mat.T)
        assert sign>0
        value*=np.exp(-log/2)
    return value


def patterns(raw,banks):
    result=np.zeros(1<<len(banks),np.int64)
    for i in range(0,len(raw),8192):
        word=(raw[i:i+8192,banks]!=0)@(1<<np.arange(len(banks)))
        result+=np.bincount(word,minlength=len(result))
    return result


def freeze():
    OUT.mkdir(exist_ok=True)
    factors=factor(*hardware())
    src=ROOT.parent/"external_sources"/"jiuzhang3_iteration112_independent"
    ref=json.loads((DATA/"iteration112_hardware_v1/sources.json").read_text())[0]
    assert ref["file"]==10 and digest(src/ref["member"])==ref["sha256"]
    raw=np.memmap(src/ref["member"],np.uint8,"r").reshape(-1,144)
    optical=np.array([1-void(factors,[i]) for i in range(144)])
    counts=np.zeros(144,int)
    for off in range(0,len(raw),8192):counts+=(raw[off:off+8192]!=0).sum(0)
    banks=np.sort(np.argsort(-abs(counts/len(raw)-optical),kind="stable")[:8])
    vacancy=np.array([void(factors,banks[(mask&(1<<np.arange(8)))!=0]) for mask in range(256)])
    word=vacancy[::-1].copy()
    for k in range(8):
        a=word.reshape(-1,2,1<<k);a[:,1]-=a[:,0]
    assert word.min()>=0 and abs(word.sum()-1)<1e-12
    train=patterns(raw,banks)/len(raw)
    event=train>word
    info=dict(training_file=10,training_sha256=ref["sha256"],banks=banks.tolist(),
              bit_convention="bit j is one if any of the eight detectors in listed bank j clicked",
              event_words=np.flatnonzero(event).tolist(),optical_probability=float(word[event].sum()),
              training_probability=float(train[event].sum()),heldout_files=[12,13],alpha_family=.01,
              confidence="One-sided bounded-difference martingale limit per file with Bonferroni correction over two files.")
    p=OUT/"hardware_protocol.json";rawbytes=(json.dumps(info,indent=2)+"\n").encode()
    if p.exists():assert p.read_bytes()==rawbytes
    else:p.write_bytes(rawbytes)
    print(banks,info["training_probability"]-info["optical_probability"],digest(p),flush=True)


def evaluate():
    from iteration112_hardware_files_v1 import run
    protocol=OUT/"hardware_protocol.json";frozen=json.loads(protocol.read_bytes())
    run(CACHE,numbers=frozen["heldout_files"])
    result=[];banks=frozen["banks"];codes=frozen["event_words"];p=frozen["optical_probability"]
    for file in json.loads((CACHE/"files.json").read_text()):
        path=CACHE/file["member"];assert digest(path)==file["sha256"]
        raw=np.memmap(path,np.uint8,"r").reshape(-1,144)
        count=int(patterns(raw,banks)[codes].sum());n=len(raw)
        excess=count/n-p
        confidence=np.sqrt(np.log(2/frozen["alpha_family"])/(2*n))
        result.append(dict(file=file["file"],sha256=file["sha256"],record_count=n,event_count=count,
                           capture_probability=count/n,finite_file_tv_lower=excess,
                           average_conditional_law_tv_lower99=excess-confidence,one_sided_radius=confidence))
    info=dict(protocol_sha256=digest(protocol),optical_probability=p,holdouts=result)
    (OUT/"hardware_holdout.json").write_bytes((json.dumps(info,indent=2)+"\n").encode())
    print(info,flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("stage",choices=("freeze","evaluate"))
    with threadpool_limits(limits=2):
        if a.parse_args().stage=="freeze":freeze()
        else:evaluate()
