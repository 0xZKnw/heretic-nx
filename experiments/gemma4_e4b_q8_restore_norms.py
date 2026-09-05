#!/usr/bin/env python3
"""Research-only Q8 row norm restoration by rescaling block scales, not codes."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from heretic_nx.edits.gguf_q8 import _copy_source, _gguf_api
from heretic_nx.edits.gguf_quant import _file_and_untouched_sha256
from heretic_nx.hashing import canonical_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gemma4-e4b-q8'


def row_norms(blocks):
    scales = np.ascontiguousarray(blocks[..., :2]).view('<f2').reshape(blocks.shape[:2])
    codes = blocks[..., 2:].view(np.int8).astype(np.float64)
    return np.sqrt(np.sum(np.square(codes).sum(axis=2) * np.square(scales.astype(np.float64)), axis=1))


def restore(base, candidate, amount):
    """Mutate Q8 scales only; return diagnostics on the actually decoded norms."""
    before = row_norms(candidate)
    reference = row_norms(base)
    if np.any((before == 0) & (reference != 0)):
        raise ValueError('cannot restore a nonzero norm by scaling a zero row')
    ratio = np.divide(reference, before, out=np.ones_like(before), where=before != 0)
    ratio = 1 + amount * (ratio - 1)
    original_codes = candidate[..., 2:].copy()
    scales = np.ascontiguousarray(candidate[..., :2]).view('<f2').reshape(candidate.shape[:2])
    rescaled = (scales.astype(np.float64) * ratio[:, None]).astype('<f2')
    if not np.isfinite(rescaled).all():
        raise ValueError('rescaling overflowed Q8 block scales')
    candidate[..., :2] = rescaled.view(np.uint8).reshape(candidate.shape[:2] + (2,))
    if not np.array_equal(original_codes, candidate[..., 2:]):
        raise AssertionError('Q8 integer codes changed')
    after = row_norms(candidate)
    denom = np.maximum(reference, 1e-30)
    return {'row_scale_min': float(ratio.min()), 'row_scale_max': float(ratio.max()),
            'max_relative_norm_error_before': float(np.max(np.abs(before-reference)/denom)),
            'max_relative_norm_error_after': float(np.max(np.abs(after-reference)/denom))}


def main():
    import gemma4_e4b_q8_build as parent

    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--amount', type=float, default=1.0)
    args = p.parse_args()
    if not 0 <= args.amount <= 1 or Path(args.tag).name != args.tag:
        raise ValueError('invalid amount or tag')
    output = ROOT / 'outputs' / f'gemma4-e4b-q8-{args.tag}.gguf'
    if output.exists():
        raise FileExistsError(output)
    prep = parent.engine._verified_preparation()
    source = args.candidate.resolve(strict=True)
    Reader, _, _, _ = _gguf_api()
    base_reader = Reader(parent.engine.BASE_Q8)
    base_tensors = {t.name: t for t in base_reader.tensors}
    names = [r['tensor_name'] for r in prep['selected']]
    intervals = tuple(sorted((int(base_tensors[n].data_offset), int(base_tensors[n].data_offset + base_tensors[n].n_bytes)) for n in names))
    source_hash = _file_and_untouched_sha256(source, intervals)
    base_hash = _file_and_untouched_sha256(parent.engine.BASE_Q8, intervals)
    if source_hash.untouched_sha256 != base_hash.untouched_sha256:
        raise ValueError('candidate changed bytes outside the declared sites')
    _copy_source(source, output)
    reader = Reader(output, mode='r+')
    tensors = {t.name: t for t in reader.tensors}
    diagnostics = []
    for name in names:
        a, b = base_tensors[name], tensors[name]
        if a.tensor_type.name != 'Q8_0' or a.tensor_type != b.tensor_type or not np.array_equal(a.shape, b.shape):
            raise ValueError(f'incompatible Q8 tensor: {name}')
        width, rows = map(int, a.shape)
        av = np.asarray(a.data).view(np.uint8).reshape(rows, width//32, 34)
        bv = np.asarray(b.data).view(np.uint8).reshape(rows, width//32, 34)
        parts = []
        for start in range(0, rows, 128):
            parts.append(restore(av[start:start+128], bv[start:start+128], args.amount))
        diagnostics.append({'tensor_name': name, 'chunks': parts})
    reader.data.flush()
    del tensors, reader, b, bv
    gc.collect()
    result_hash = _file_and_untouched_sha256(output, intervals)
    if result_hash.untouched_sha256 != base_hash.untouched_sha256:
        raise AssertionError('untouched bytes changed')
    report = {'method': 'rescale_q8_block_scales_preserving_integer_codes',
              'amount': args.amount, 'source': str(source), 'source_sha256': source_hash.sha256,
              'output': str(output), 'output_sha256': result_hash.sha256,
              'untouched_sha256': result_hash.untouched_sha256, 'diagnostics': diagnostics}
    (RUN / f'{args.tag}.build.json').write_bytes(canonical_json(report) + b'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'diagnostics'}), flush=True)


if __name__ == '__main__':
    main()
