"""Evaluate supplied GBS evidence or run a fixed positive/negative demonstration.

Input NPZ uses covariance plus optional distribution, samples, and
sample_log_probabilities. A native wrapper JSON can be supplied instead of
those evidence arrays. Statistical claims require a prespecified plan and an
explicit fixed-first-batch protocol; this CLI does not silently attest them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from gbslab.cpp_click import run_cpp_click
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.fock import threshold_from_fock
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.gbs_identity import (
    ConfidencePlan, assess_gbs_identity, assess_native_samples,
    covariance_sha256, default_witness_subsets, fixed_batch_protocol,
)
from gbslab.paired_stress import dense_target, paired_covariance


SCHEMA={
    "schema_version":1,
    "purpose":"Agreement with one specified zero-mean Gaussian threshold target; not a generic Gaussian-histogram classifier",
    "decisions":["accepted_within_tolerance_conditional","rejected","insufficient_evidence",
                 "invalid_evidence","invalid_model","evaluation_failed","out_of_scope"],
    "evidence_scopes":["full_distribution","global_tv_upper","witnesses_only","none"],
    "target_binding":["target.target_id","target.covariance_sha256","target.detector_model",
                      "target.state_model","target.modes","target.covariance_convention"],
    "native_sampler_binding":["native_provenance.executable_sha256","native_provenance.selector",
                              "native_provenance.k","native_provenance.order","native_provenance.vacuum_kernel"],
    "quality_fields":["full_law.tv","global_tv.upper","witnesses.tv_lower","epsilon"],
    "qualifications":["strict_floating_point_certificate=false","sampling_protocol","confidence_plan",
                      "global_tv.output_space=binary_patterns_plus_failure",
                      "global_tv.conditional_on_success_coverage_claimed=false"],
    "trust_boundary":"Metadata hashes bind records; they are not cryptographic proof that an arbitrary external sampler is honest or correct",
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_output(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Use a fresh or empty output directory")
    path.mkdir(parents=True,exist_ok=True)
    return path


def snapshot(output):
    paths=[Path(__file__),ROOT/"src/gbslab/gbs_identity.py",ROOT/"src/gbslab/cpp_click.py",
           ROOT/"src/gbslab/exact.py",ROOT/"src/gbslab/schur_memory.py",ROOT/"src/gbslab/tv_validation.py",
           ROOT/"src/gbslab/fock.py",ROOT/"src/gbslab/gaussian.py",ROOT/"src/gbslab/paired_stress.py"]
    directory=output/"source_snapshot";directory.mkdir()
    manifest=[]
    for path in paths:
        shutil.copy2(path,directory/path.name)
        manifest.append({"source":str(path.relative_to(ROOT)),"snapshot":"source_snapshot/"+path.name,
                         "sha256":digest(path)})
    (output/"schema.json").write_text(json.dumps(SCHEMA,indent=2)+"\n")
    return manifest


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def run_input(args):
    output=fresh_output(args.output)
    sources=snapshot(output)
    with np.load(args.input,allow_pickle=False) as archive:
        covariance=np.array(archive["covariance"],copy=True)
        arrays={key:np.array(archive[key],copy=True) for key in
                ("distribution","samples","sample_log_probabilities") if key in archive}
    plan=None
    if args.plan:
        data=json.loads(args.plan.read_text())
        plan=ConfidencePlan(tuple(data["comparison_ids"]),data["familywise_alpha"])
    protocol=json.loads(args.protocol.read_text()) if args.protocol else None
    options={"plan":plan,"comparison_id":args.comparison_id,"epsilon":args.epsilon,
             "max_exact_clicks":args.max_exact_clicks,"protocol":protocol,
             "detector_model":args.detector_model,"target_id":args.target_id,
             "keep_scores":True}
    if args.native_result:
        native=json.loads(args.native_result.read_text())
        result=assess_native_samples(covariance,native,**options)
    else:
        result=assess_gbs_identity(covariance,candidate_distribution=arrays.get("distribution"),
            samples=arrays.get("samples"),sample_log_probabilities=arrays.get("sample_log_probabilities"),**options)
    write_json(output/"assessment.json",result)
    references={"input":{"path":str(args.input.resolve()),"sha256":digest(args.input)}}
    for key,path in [("plan",args.plan),("protocol",args.protocol),("native_result",args.native_result)]:
        if path:references[key]={"path":str(path.resolve()),"sha256":digest(path)}
    write_json(output/"manifest.json",{"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
        "sources":sources,"inputs":references,"assessment_sha256":digest(output/"assessment.json"),
        "decision":result["decision"],"numpy":np.__version__})
    print(json.dumps({"decision":result["decision"],"scope":result["evidence_scope"],"output":str(output)}))


def run_demo(args):
    output=fresh_output(args.output)
    sources=snapshot(output)
    ids=("full_tmsv","iid_tmsv","matched_means_impostor","witness_only_tmsv",
         "unresolved_target","wrong_detector","independent_fock","native_complete")
    plan=ConfidencePlan(ids,.05)
    config={"comparison_ids":list(ids),"familywise_alpha":.05,"samples_per_iid_case":2048,
            "epsilon":.1,"max_exact_clicks":2,"sample_seeds":{"tmsv":12301,"impostor":12302},
            "native_seed":12304,"native_samples":256,"native_kernel":args.vacuum_kernel,
            "protocol":"first fixed batch, iid PRNG model, no retry/filtering; one batch reused for positive and witness-only diagnostics",
            "role":"small positive/negative controls for evaluator semantics, not a new GBS algorithm performance claim"}
    # Persist the family and sampling schedule before any draw.
    write_json(output/"protocol.json",config)
    manifest={"status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
              "confidence_plan":plan.as_dict(),"sources":sources,"cases":[]}
    write_json(output/"manifest.json",manifest)
    V=paired_covariance(1,.5);p=dense_target(1,.5)
    n=config["samples_per_iid_case"]
    pair=np.random.default_rng(config["sample_seeds"]["tmsv"]).integers(0,2,size=n,dtype=np.uint8)
    correct=np.column_stack([pair,pair])
    false=np.random.default_rng(config["sample_seeds"]["impostor"]).integers(0,2,size=(n,2),dtype=np.uint8)
    records=[]

    def save(name,covariance,assessment,*,arrays=None,expected=None,notes=None):
        data={"covariance":covariance,**(arrays or {})}
        path=output/(name+".npz");np.savez_compressed(path,**data)
        report=output/(name+".json");write_json(report,assessment)
        if expected is not None and assessment["decision"]!=expected:
            raise AssertionError(f"Unexpected evaluator decision for {name}: {assessment['decision']}")
        row={"case":name,"decision":assessment["decision"],"expected_decision":expected,
             "evidence_scope":assessment["evidence_scope"],"input":path.name,"input_sha256":digest(path),
             "assessment":report.name,"assessment_sha256":digest(report),
             "covariance_sha256":covariance_sha256(covariance),"notes":notes}
        records.append(row);manifest["cases"]=records;write_json(output/"manifest.json",manifest)
        print(name,assessment["decision"],flush=True)

    save("full_tmsv",V,assess_gbs_identity(V,candidate_distribution=p,epsilon=.1,
         plan=plan,comparison_id="full_tmsv",target_id="analytic_two_mode_squeezed_vacuum_p0.5"),
         arrays={"distribution":p},expected="accepted_within_tolerance_conditional")
    common={"epsilon":.1,"plan":plan,"max_exact_clicks":2,
            "protocol":fixed_batch_protocol(n,source="saved_demo_protocol"),"keep_scores":True}
    save("iid_tmsv",V,assess_gbs_identity(V,samples=correct,
         sample_log_probabilities=np.full(n,-math.log(2)),comparison_id="iid_tmsv",**common),
         arrays={"samples":correct,"sample_log_probabilities":np.full(n,-math.log(2)),"analytic_target":p},
         expected="accepted_within_tolerance_conditional")
    save("matched_means_impostor",V,assess_gbs_identity(V,samples=false,
         sample_log_probabilities=np.full(n,-2*math.log(2)),comparison_id="matched_means_impostor",**common),
         arrays={"samples":false,"sample_log_probabilities":np.full(n,-2*math.log(2)),"analytic_target":p,
                 "analytic_sampler_distribution":np.full(4,.25)},expected="rejected",
         notes="Independent thermal clicks are themselves a different Gaussian target. Rejection concerns the specified correlated TMSV target; single-mode theoretical means match exactly.")
    save("witness_only_tmsv",V,assess_gbs_identity(V,samples=correct,
         comparison_id="witness_only_tmsv",**common),arrays={"samples":correct},
         expected="insufficient_evidence")
    def unresolved(sample):raise NumericalProbabilityError("deliberate unresolved oracle control")
    save("unresolved_target",V,assess_gbs_identity(V,samples=correct,
         sample_log_probabilities=np.full(n,-math.log(2)),comparison_id="unresolved_target",
         target_log_probability=unresolved,target_probability_mode="deliberately_unresolved_control",**common),
         arrays={"samples":correct,"sample_log_probabilities":np.full(n,-math.log(2))},
         expected="insufficient_evidence")
    save("wrong_detector",V,assess_gbs_identity(V,detector_model="pnr",plan=plan,
         comparison_id="wrong_detector"),expected="out_of_scope",
         notes="PNR GBS is not declared invalid; it is outside the threshold evaluator.")
    U=haar_unitary(2,np.random.default_rng(12303));r=np.array([.25,.35]);eta=.7
    fock=threshold_from_fock(U,r,eta,max_photons=12)
    VF=covariance_from_interferometer(U,r,eta)
    assessment=assess_gbs_identity(VF,candidate_distribution=fock.normalized_distribution,
         epsilon=1e-5,plan=plan,comparison_id="independent_fock")
    assessment["independent_fock_reference"]={"omitted_mass":fock.omitted_mass,"retained_mass":fock.retained_mass,
         "max_photons":fock.max_photons,"states_enumerated":fock.fock_states_enumerated,
         "method":"creation-operator amplitude recurrence, loss thinning, and threshold aggregation; no covariance determinant oracle"}
    if assessment["full_law"]["tv"]>fock.omitted_mass+1e-12:raise AssertionError("Fock truncation bound mismatch")
    save("independent_fock",VF,assessment,arrays={"U":U,"squeezing":r,"transmissivity":np.array(eta),
        "distribution":fock.normalized_distribution,"fock_subprobability":fock.probabilities},
        expected="accepted_within_tolerance_conditional")
    if args.executable:
        VN=covariance_from_interferometer(haar_unitary(3,np.random.default_rng(12305)),.55,.65)
        native=run_cpp_click(VN,selection="click_influence",max_clicks=2,samples=256,seed=12304,
            raw_samples=True,record_log_probabilities=True,vacuum_kernel=args.vacuum_kernel,
            enumerate_distribution=True,executable=args.executable)
        write_json(output/"native_complete_raw.json",native)
        assessment=assess_native_samples(VN,native,epsilon=.01,plan=plan,
            comparison_id="native_complete",protocol=fixed_batch_protocol(256,source="saved_demo_protocol"),
            max_exact_clicks=3,keep_scores=True)
        save("native_complete",VN,assessment,arrays={"samples":np.asarray(native["raw_samples"],dtype=np.uint8),
             "sample_log_probabilities":np.asarray(native["sample_log_probabilities"]),
             "distribution":np.asarray(native["distribution"])},expected="accepted_within_tolerance_conditional")
    manifest.update(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
                    completed_cases=len(records),planned_cases=len(ids),
                    native_demo_skipped=args.executable is None,numpy=np.__version__)
    write_json(output/"manifest.json",manifest)
    print(json.dumps({"status":"complete","cases":len(records),"output":str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",type=Path,help="NPZ with covariance and optional evidence arrays")
    mode.add_argument("--demo",action="store_true",help="fixed small positive/negative calibration controls")
    mode.add_argument("--describe-schema",action="store_true")
    parser.add_argument("--output",type=Path,help="fresh or empty output directory")
    parser.add_argument("--native-result",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--comparison-id")
    parser.add_argument("--epsilon",type=float,default=.1)
    parser.add_argument("--max-exact-clicks",type=int,choices=range(21),default=6)
    parser.add_argument("--detector-model",default="threshold")
    parser.add_argument("--target-id")
    parser.add_argument("--executable",type=Path,help="opt-in native demonstration executable")
    parser.add_argument("--vacuum-kernel",choices=["log","product"],default="log")
    args=parser.parse_args()
    if args.describe_schema:
        print(json.dumps(SCHEMA,indent=2));return
    if args.output is None:parser.error("--output is required")
    if args.demo:
        if args.plan or args.protocol or args.comparison_id or args.native_result:
            parser.error("the demo owns its fixed plan/protocol; external evidence options cannot override it")
        run_demo(args)
    else:run_input(args)


if __name__=="__main__":main()
