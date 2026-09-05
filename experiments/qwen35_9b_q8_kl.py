#!/usr/bin/env python3
"""Full padded-vocabulary KL for Qwen3.5-9B; never truncate to tokenizer len."""
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from heretic_nx.eval.native_logits import attest_tokenizer_assets
from heretic_nx.hashing import sha256_json
import gemma4_e2b_q8_kl as engine
import qwen35_9b_q8_eval as refusal

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 248320
TOKENIZER_SIZE = 248077


def validate_vocabulary(config, tokenizer_size):
    if config["text_config"]["vocab_size"] != VOCAB_SIZE or tokenizer_size != TOKENIZER_SIZE:
        raise RuntimeError("unexpected pinned Qwen tokenizer/output vocabulary")


def prompts():
    path = refusal.engine.TOKENIZER_PATH
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    config = json.loads((path / "config.json").read_text())
    validate_vocabulary(config, len(tokenizer))
    rows = load_dataset(engine.GOOD_DATASET, revision=engine.GOOD_REVISION, split="test")
    rendered = refusal.engine.render(tokenizer, [str(rows[i]["text"]) for i in range(104)])
    tokens = [tokenizer.encode(text, add_special_tokens=False) for text in rendered]
    # The collector's vocabulary field denotes the complete output-logit domain,
    # including Qwen's padded rows, not the number of named tokenizer entries.
    identity = attest_tokenizer_assets(
        path, vocab_size=VOCAB_SIZE,
        tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    )
    return tokens, sha256_json(tokens), identity


engine.RUN_DIR = ROOT / "runs/qwen35-9b-q8/kl"
engine.VOCAB_SIZE = VOCAB_SIZE
engine.RAW_LOGIT_SCHEMA = "qwen35-9b-q8-first-token-raw-logits-v1"
engine.SCHEMA_FAMILY = "qwen35-9b"
engine.prompts = prompts

if __name__ == "__main__":
    engine.main()
