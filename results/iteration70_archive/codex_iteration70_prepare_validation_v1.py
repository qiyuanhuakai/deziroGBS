from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'.git/iteration70_pgf_quad_v1.cpp';target=ROOT/'native/gaussian_pgf_quad.cpp';assert not target.exists();target.write_bytes(source.read_bytes())
s=(ROOT/'.git/codex_iteration70_pgf_quad_v1.py').read_text(encoding='utf-8')
s=s.replace("OUT=ROOT/'.git/iteration70_pgf_quad_v1_20261110';OUT.mkdir(exist_ok=False)", "import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)")
s=s.replace("ROOT/'.git/iteration70_pgf_quad_v1.cpp'", "ROOT/'native/gaussian_pgf_quad.cpp'")
p=ROOT/'experiments/iteration70_pgf_reference.py';assert not p.exists();compile(s,str(p),'exec');p.write_bytes(s.encode())
s=(ROOT/'experiments/iteration69_tests.py').read_text(encoding='utf-8').replace('from iteration69_build import','from iteration70_build import')
s=s.replace("parser.add_argument('--reset-build',type=Path,required=True);", "parser.add_argument('--reset-build',type=Path,required=True);parser.add_argument('--dual-build',type=Path,required=True);")
s=s.replace("    sources=sorted", "    os.environ['GBSLAB_GAUSSIAN_RESET_PARALLEL_LIBRARY']=str(args.dual_build.resolve()/'gaussian_reset_parallel.dll');os.environ['GBSLAB_DIRECT_CLICK_LIBRARY']=str(args.dual_build.resolve()/'gaussian_direct_click.dll')\n    sources=sorted")
s=s.replace("'GBSLAB_WIRE_FOREST_LIBRARY')", "'GBSLAB_WIRE_FOREST_LIBRARY','GBSLAB_GAUSSIAN_RESET_PARALLEL_LIBRARY','GBSLAB_DIRECT_CLICK_LIBRARY')")
s=s.replace("report=dict(tests_run=", "report=dict(parallel_tests_passed=sum(s.startswith('test_native_gaussian_reset_parallel.') for s in result.successes),direct_click_tests_passed=sum(s.startswith('test_native_direct_click.') for s in result.successes),tests_run=")
p=ROOT/'experiments/iteration70_tests.py';assert not p.exists();compile(s,str(p),'exec');p.write_bytes(s.encode())
