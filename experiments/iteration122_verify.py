"""Verify archived decisions, delivered rows, exact family tails and file digests."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from iteration122_powers import ROOT, OUT, json_write
from iteration122_family_certificate import uniform


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(case, route, first, domain):
    folder = OUT / case
    key = 'click' if route else 'pnr'
    base = folder / f'base_{route}_128.npz'
    result = folder / f'full_{route}_128.npz'
    meta = json.loads(base.with_suffix('.json').read_bytes())
    report = json.loads(result.with_suffix('.json').read_bytes())
    assert meta['first_proposal'] == first
    selected = []
    assert len(report['decisions']) == 128
    for i, row in enumerate(report['decisions']):
        t = 4 * np.random.default_rng([domain, 92, 0, route, i]).random()
        assert row['threshold'] == t and row['index'] == i
        w = row['weights']
        assert len(w) == row['stages']
        if t < .5:
            assert not w
            accepted = True
        elif t < .8 * w[0] or t >= 1.25 * w[0]:
            assert len(w) == 1
            accepted = t < .8 * w[0]
        else:
            assert len(w) == 2
            accepted = t < w[1]
        assert accepted == row['accepted']
        if accepted:
            selected.append(i)
    assert len(selected) == report['accepted']
    assert np.isclose((meta['wall_seconds'] + report['scoring_wall_seconds']) / len(selected),
                      report['seconds_per_accepted'])
    with np.load(base) as raw, np.load(result) as out:
        for spec in (0, 1):
            log = raw[f'log{route}_{spec}']
            assert np.array_equal(log[:, 0], np.arange(first, first + len(log)))
            assert len(log) < 2**22
            draws = [np.random.default_rng([119, 82, route, spec, int(index)]).random()
                     for index in log[:, 0]]
            assert np.array_equal(log[:, 3], draws)
            assert int((log[:, 3] < log[:, 2]).sum()) == meta['spectra'][spec]['successful']
        a, b = [raw[f'row{route}_{spec}'] for spec in (0, 1)]
        combined = a | b if route else np.add(a, b, dtype=np.uint32)
        assert np.array_equal(combined, raw[key])
        assert np.array_equal(out['accepted_indices'], selected)
        assert np.array_equal(out[key], combined[selected])
        if case == 'dominated':
            with np.load(folder / f'{key}.npz') as obs:
                assert np.array_equal(obs['records'], out[key][:, :96])


def files():
    paper = ROOT / 'papers/number-tilted-gbs'
    with zipfile.ZipFile(paper / 'latex-source.zip') as src:
        assert src.testzip() is None
        assets = [paper / name for name in src.namelist()]
        assert all(p.read_bytes() == src.read(p.name) for p in assets)
    deps = ['iteration119_mixture', 'iteration119_clamps', 'iteration120_generate',
            'iteration120_replay', 'iteration120_interval']
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json']
    paths += list((ROOT / 'experiments').glob('iteration122_*.py'))
    paths += [ROOT / 'experiments' / (name + '.py') for name in deps]
    paths += [ROOT / 'native/bright_quartic122.cpp', paper / 'manuscript.pdf',
              paper / 'latex-source.zip', paper / 'README.md', *assets]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def main(write):
    for case, block, domain in [('low', 4, 1220), ('median', 6, 1221), ('dominated', 12, 1229)]:
        for route in (0, 1):
            check_run(case, route, (block + route) * 2**22, domain)
    assert uniform() == json.loads((OUT / 'families.json').read_bytes())['isotropic']
    for case in ('low', 'median', 'high'):
        cert = json.loads((OUT / case / 'certificate.json').read_bytes())
        assert digest(OUT / case / 'input.npz') == cert['input_sha256']
        assert max(map(float, cert['signed_upper'])) < float(cert['full_tv_upper'])
    for case in ('unequal_loss', 'bright'):
        info = json.loads((OUT / case / 'records.json').read_bytes())
        assert digest(OUT / case / 'input.npz') == info['input_sha256']
        with np.load(OUT / case / 'records.npz') as records:
            for row in info['readouts']:
                a = records[row['route']]
                assert a.shape == (4096, info['visible_ports']) and int(a.max()) == row['maximum_port']
                assert np.isclose(sum(row['seconds']) * 1e6 / len(a), row['microseconds_per_delivered'])
    path = OUT / 'manifest.json'
    inventory = files()
    if write:
        json_write(path, inventory)
    assert json.loads(path.read_bytes()) == inventory
    print(f'{len(inventory)} digests and 768 final decisions verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)
