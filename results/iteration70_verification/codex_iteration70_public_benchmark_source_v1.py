from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'.git/codex_iteration70_dual_benchmark_v1.py').read_text(encoding='utf-8')
s=s.replace("OUT=ROOT/'.git/iteration70_dual_benchmark_v1_20261110';OUT.mkdir(exist_ok=False)", "import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)")
s=s.replace("ROOT/'.git/iteration70_frugal_large_v1_20261110'", "ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110'")
s=s.replace("ROOT/'.git/iteration70_direct_click_v1.cpp'", "ROOT/'native/gaussian_direct_click.cpp'")
p=ROOT/'experiments/iteration70_benchmark.py';assert not p.exists();compile(s,str(p),'exec');p.write_bytes(s.encode())
