"""Run experiment imports in isolated processes: adapters configure shared engines."""
import os
from pathlib import Path
import subprocess
import sys


def test_qwen_padded_domain_and_independent_requests():
    root = Path(__file__).resolve().parents[1]
    code = '''
from qwen35_9b_q8_kl import validate_vocabulary, VOCAB_SIZE
from qwen35_9b_q8_eval import IndependentPromptClient
validate_vocabulary({"text_config": {"vocab_size": 248320}}, 248077)
assert VOCAB_SIZE == 248320
for config, size in [({"text_config": {"vocab_size": 248077}}, 248077),
                     ({"text_config": {"vocab_size": 248320}}, 248076)]:
    try:
        validate_vocabulary(config, size)
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted a changed or truncated vocabulary")
class HTTP:
    def request_json(self, path, **kwargs):
        assert path == "/completion"
        assert kwargs["payload"] == {"prompt": [1,2], "n_predict": 96,
            "temperature": -1, "stream": False, "cache_prompt": False}
        return {"content": "ok"}
c = IndependentPromptClient("http://127.0.0.1:1236")
c._http = HTTP()
assert c.completion([1,2], max_tokens=96) == "ok"
'''
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "experiments"), str(root / "src")))}
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True, timeout=90)
