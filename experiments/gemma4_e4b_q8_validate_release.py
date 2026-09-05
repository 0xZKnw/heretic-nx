#!/usr/bin/env python3
"""Recheck natural-order refusal and paired capabilities after both search gates."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from heretic_nx.hashing import sha256_file
from gemma4_e4b_q8_screen_candidate import refusal_passed

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gemma4-e4b-q8'
BASE = ROOT / 'checkpoints/gemma4-e4b-gguf/gemma-4-E4B-it-Q8_0.gguf'


@contextmanager
def server(artifact, alias, resume=False):
    with (RUN / f'{alias}.server.log').open('a' if resume else 'x') as log:
        proc = subprocess.Popen([
            str(ROOT / 'build/llama.cpp-native/bin/llama-server'),
            '-m', str(artifact), '--alias', alias, '--port', '1236',
            '-ngl', '99', '-c', '8192', '-np', '4', '-lv', '1',
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited: {alias}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:1236/health', timeout=1) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except Exception:
                    time.sleep(.25)
            else:
                raise TimeoutError(alias)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run(script, *args):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))}
    subprocess.run([sys.executable, str(ROOT / 'experiments' / script), *map(str, args)], check=True, env=env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--search-label', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    artifact = args.artifact.resolve(strict=True)
    digest = sha256_file(artifact)
    refusal = json.loads((RUN / 'refusal' / f'{args.search_label}.json').read_text())
    kl = json.loads((RUN / 'kl' / f'{args.search_label}-vs-base.json').read_text())
    if (not refusal_passed(refusal) or not kl['passed'] or kl['count'] != 104
            or not 0 <= kl['mean_first_token_kl'] <= .05
            or refusal['runtime_model']['artifact_sha256'] != digest
            or kl['candidate_artifact']['sha256'] != digest
            or kl['base_artifact']['sha256'] != sha256_file(BASE)):
        raise ValueError('candidate does not have matching, passing full search gates')
    label = args.search_label + '-final'
    final_path = RUN / 'refusal' / f'{label}.json'
    with server(artifact, label, args.resume):
        if not args.resume or not final_path.is_file():
            run('gemma4_e4b_q8_eval.py', '--artifact', artifact, '--model', label,
                '--label', label, '--row-count', 104, '--max-new-tokens', 96)
        final = json.loads(final_path.read_text())
        if (not refusal_passed(final) or final['row_indices'] != list(range(104))
                or final['runtime_model']['artifact_sha256'] != digest):
            raise ValueError('natural-order refusal recheck failed; capability skipped')
        run('gemma4_e4b_q8_capability.py', 'collect', '--artifact', artifact,
            '--model', label, '--label', label)
    with server(BASE, 'gemma4-e4b-base-q8-capability', args.resume):
        run('gemma4_e4b_q8_capability.py', 'collect', '--artifact', BASE,
            '--model', 'gemma4-e4b-base-q8-capability', '--label', 'base-q8')
    run('gemma4_e4b_q8_capability.py', 'compare', '--base-label', 'base-q8',
        '--candidate-label', label)


if __name__ == '__main__':
    main()
