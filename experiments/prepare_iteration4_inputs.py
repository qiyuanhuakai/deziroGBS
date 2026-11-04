"""Freeze new synthetic cross-mode inputs before timing either native kernel."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import numpy as np

from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.exact import ThresholdModel

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args):
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    if len(config["modes"])*len(config["interferometer_seeds"])*len(config["profiles"])!=config["validation"]["planned_comparisons"]:
        raise ValueError("Planned comparison count mismatch")
    shutil.copy2(args.config,args.output/"config.json")
    snapshot=args.output/"source_snapshot";snapshot.mkdir()
    for p in [Path(__file__),ROOT/"src/gbslab/gaussian.py",ROOT/"src/gbslab/exact.py"]:shutil.copy2(p,snapshot/p.name)
    cases=[]
    for m in config["modes"]:
        for seed in config["interferometer_seeds"]:
            effective_seed=seed+1009*m
            U=haar_unitary(m,np.random.default_rng(effective_seed))
            for profile in config["profiles"]:
                spec={"modes":m,"interferometer_seed":seed,"effective_interferometer_seed":effective_seed,**profile}
                case_id=f"m{m}_{profile['name']}_s{seed}"
                V=covariance_from_interferometer(U,profile["squeezing"],profile["transmissivity"])
                arrays={"covariance":V,"interferometer":U,"squeezing":np.full(m,profile["squeezing"]),
                        "transmissivity":np.array(profile["transmissivity"])}
                if m<=config["validation"]["full_target_max_modes"]:
                    arrays["target"]=ThresholdModel(V).full_distribution()
                # Exact single-mode Gaussian marginals, not a complete-law quality claim.
                p0=[]
                for j in range(m):
                    ids=[j,j+m]
                    p0.append(2/np.sqrt(np.linalg.det(V[np.ix_(ids,ids)]+np.eye(2))))
                path=args.output/f"{case_id}.npz";np.savez_compressed(path,**arrays)
                cases.append({"case_id":case_id,"spec":spec,"input":str(path.relative_to(ROOT)),"sha256":sha(path),
                              "physical_target_mean_clicks":float(m-np.sum(p0)),
                              "full_target_available":"target" in arrays,
                              "source":"synthetic_fixed_seed_not_experimental_data"})
    manifest={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha(args.config),
              "cases":cases,"case_count":len(cases),"interferometer_count":len(config["modes"])*len(config["interferometer_seeds"]),
              "source_hashes":{p.name:sha(p) for p in snapshot.iterdir()},"numpy":np.__version__}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"cases":len(cases),"interferometers":manifest["interferometer_count"],"output":str(args.output)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/iteration4_cross_mode.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/iteration4_inputs")
    run(p.parse_args())
