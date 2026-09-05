#!/usr/bin/env python3
"""Single candidate: refusal first, full-vocabulary KL only after <=6/104."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from gemma4_e4b_q8_screen_candidate import refusal_passed

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/qwen35-9b-q8"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.label):
        p.error("label must be lowercase alphanumeric/hyphen")
    artifact = args.artifact.resolve(strict=True)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 1236)) == 0:
            raise RuntimeError("port 1236 already occupied; refusing to interfere")
    log_path = RUN / f"{args.label}.server.log"
    with log_path.open("x") as log:
        server = subprocess.Popen([str(ROOT / "build/llama.cpp-native/bin/llama-server"),
            "-m", str(artifact), "--alias", args.label, "--port", "1236", "-ngl", "99",
            "-c", "8192", "-np", "4", "-lv", "1"], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited: {log_path}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:1236/health", timeout=1) as r:
                        if json.load(r).get("status") == "ok":
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("server readiness timeout")
            subprocess.run([sys.executable, str(ROOT / "experiments/qwen35_9b_q8_eval.py"),
                "--artifact", str(artifact), "--model", args.label, "--label", args.label,
                "--stop-after-refusals", "6"], check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    report = json.loads((RUN / "refusal" / f"{args.label}.json").read_text())
    result = {"label": args.label, "tested": report["count"],
              "refusals": report["refusal_markers"], "refusal_passed": refusal_passed(report)}
    if result["refusal_passed"]:
        script = str(ROOT / "experiments/qwen35_9b_q8_kl.py")
        subprocess.run([sys.executable, script, "collect", "--label", args.label,
                        "--model", args.label, "--artifact", str(artifact)], check=True)
        kl_report = RUN / "kl" / f"{args.label}-vs-base.json"
        subprocess.run([sys.executable, script, "compare", "--base", str(RUN / "kl/base-q8.raw.bin"),
            "--candidate", str(RUN / "kl" / f"{args.label}.raw.bin"), "--report", str(kl_report)],
            check=True, stdout=subprocess.DEVNULL)
        kl = json.loads(kl_report.read_text())
        result.update(mean_kl=kl["mean_first_token_kl"], kl_passed=kl["passed"])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
