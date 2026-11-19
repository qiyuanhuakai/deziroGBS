"""Canonical array fingerprints and JSON records for iteration 23."""
import hashlib
import json
from pathlib import Path
import numpy as np


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def mps_fingerprint(tensors):
    value=hashlib.sha256()
    for i,a in enumerate(tensors):
        a=np.ascontiguousarray(a)
        value.update(json.dumps(dict(site=i,shape=list(a.shape),dtype=a.dtype.str),sort_keys=True,separators=(',',':')).encode())
        value.update(b'\0'); value.update(memoryview(a).cast('B'))
    return value.hexdigest()


def load_mps(path):
    with np.load(path,allow_pickle=False) as data:
        return [data[f'A{i}'] for i in range(len(data.files))]
