#!/usr/bin/env python3
"""Blend final-position and full-trajectory benign protection for Q8."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from heretic_nx.edits import GGUFQuantizedAblationPlan, GGUFQuantizedTensorEdit, apply_quantized_gguf_ablation
from heretic_nx.hashing import canonical_json, sha256_file
import gemma4_e4b_q8_build as parent

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gemma4-e4b-q8'
LAST = RUN / 'coord-nearmiss-native-b104-protected-s1p1-factors.safetensors'
ALL = RUN / 'coord-nearmiss-native-b2-protected-s1p1-allpositions-factors.safetensors'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blend', type=float, required=True)
    p.add_argument('--sites', default='0,1,2,3,4,5,6',
                   help='Sites receiving trajectory protection; others keep the last-position factors.')
    p.add_argument('--scale', type=float, default=1.0)
    args = p.parse_args()
    if not 0 <= args.blend <= 1:
        raise ValueError('blend must be between zero and one')
    sites = [int(s) for s in args.sites.split(',')]
    if len(sites) != len(set(sites)) or any(s < 0 or s > 6 for s in sites):
        raise ValueError('sites must be unique indices from zero to six')
    if not 0 < args.scale < float('inf'):
        raise ValueError('scale must be finite and positive')
    tag = 'trajectory-blend-' + f'{args.blend:g}'.replace('.', 'p')
    if set(sites) != set(range(7)):
        tag += '-sites' + ''.join(map(str, sorted(sites)))
    if args.scale != 1:
        tag += '-scale' + f'{args.scale:g}'.replace('.', 'p')
    a, b = load_file(LAST), load_file(ALL)
    payload, edits = {}, []
    for i, row in enumerate(parent.engine._verified_preparation()['selected']):
        axis, right = f'site{i:02d}.axis', f'site{i:02d}.native_right'
        if not torch.equal(a[axis], b[axis]):
            raise RuntimeError('blend requires identical left factors')
        payload[axis] = a[axis].contiguous()
        blend = args.blend if i in sites else 0.0
        payload[right] = (args.scale * ((1 - blend) * a[right] + blend * b[right])).contiguous()
        edits.append(GGUFQuantizedTensorEdit(
            tensor_name=row['tensor_name'], expected_quantization='Q8_0',
            a_key=axis, right_key=right, strength=1, preserve_row_norms=False,
            preserve_original_blocks=True))
    factors = RUN / f'{tag}.safetensors'
    save_file(payload, factors, metadata={'source_last_sha256': sha256_file(LAST),
                                        'source_all_sha256': sha256_file(ALL),
                                        'blend': str(args.blend),
                                        'sites': ','.join(map(str, sorted(sites))),
                                        'scale': str(args.scale)})
    plan = GGUFQuantizedAblationPlan(source_sha256=sha256_file(parent.engine.BASE_Q8),
        tensor_artifact_sha256=sha256_file(factors), edits=tuple(edits),
        row_chunk_size=256, verify_untouched_bytes=True)
    plan_path = RUN / f'{tag}.plan.json'
    plan.write(plan_path)
    merge = apply_quantized_gguf_ablation(parent.engine.BASE_Q8,
        ROOT / 'outputs' / f'gemma4-e4b-q8-{tag}.gguf', plan_path, factors)
    (RUN / f'{tag}.build.json').write_bytes(canonical_json({
        'blend': args.blend, 'sites': sorted(sites), 'scale': args.scale,
        'merge': merge}) + b'\n')
    print(json.dumps({'label': tag, 'output': merge['output']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
