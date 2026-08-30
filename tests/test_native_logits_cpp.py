from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = os.environ.get("HERETIC_NX_NATIVE_COLLECTOR")
LLAMA_CPP_SOURCE = os.environ.get("HERETIC_NX_LLAMA_CPP_SOURCE")
RUNTIME_DIR = os.environ.get("HERETIC_NX_NATIVE_RUNTIME_DIR")

pytestmark = pytest.mark.skipif(
    not (COLLECTOR and LLAMA_CPP_SOURCE and RUNTIME_DIR),
    reason="native llama.cpp build paths are not configured",
)


def _configured_paths() -> tuple[Path, Path, Path]:
    collector = Path(str(COLLECTOR)).resolve(strict=True)
    source = Path(str(LLAMA_CPP_SOURCE)).resolve(strict=True)
    runtime = Path(str(RUNTIME_DIR)).resolve(strict=True)
    return collector, source, runtime


def test_native_collector_smoke_usage() -> None:
    collector, _source, _runtime = _configured_paths()

    completed = subprocess.run(
        [str(collector)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "usage: llama_raw_logits" in completed.stderr


def test_native_collector_has_relative_runtime_search_path() -> None:
    collector, _source, runtime = _configured_paths()
    if sys.platform == "darwin":
        tool = shutil.which("otool")
        assert tool is not None
        binaries = (collector, runtime / "libllama.dylib", runtime / "libggml.dylib")
        for binary in binaries:
            completed = subprocess.run(
                [tool, "-l", str(binary.resolve(strict=True))],
                check=True,
                capture_output=True,
                text=True,
            )
            rpaths = re.findall(
                r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)",
                completed.stdout,
            )
            assert rpaths == ["@loader_path"]
    elif sys.platform.startswith("linux"):
        tool = shutil.which("readelf")
        assert tool is not None
        binaries = (collector, runtime / "libllama.so", runtime / "libggml.so")
        for binary in binaries:
            completed = subprocess.run(
                [tool, "-d", str(binary.resolve(strict=True))],
                check=True,
                capture_output=True,
                text=True,
            )
            runtime_paths = re.findall(
                r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", completed.stdout
            )
            assert runtime_paths == ["$ORIGIN"]
    else:
        pytest.skip("relative loader-path assertion is defined for macOS/Linux")


def test_exclusive_output_never_unlinks_paths(tmp_path: Path) -> None:
    _collector, source, runtime = _configured_paths()
    harness_source = tmp_path / "exclusive_output_harness.cpp"
    harness = tmp_path / "exclusive_output_harness"
    collector_source = ROOT / "experiments" / "llama_raw_logits.cpp"
    harness_source.write_text(
        "#define main llama_raw_logits_collector_main\n"
        f'#include "{collector_source.as_posix()}"\n'
        "#undef main\n"
        "int main(int argc, char ** argv) {\n"
        "    if (argc == 2 && std::string(argv[1]) == \"--inventory\") {\n"
        "        std::vector<std::string> modules;\n"
        "        std::string method;\n"
        "        if (!collect_loaded_modules(modules, method)) return 4;\n"
        "        std::cout << method << '\\n';\n"
        "        for (const auto & module : modules) std::cout << module << '\\n';\n"
        "        return 0;\n"
        "    }\n"
        "    if (argc == 3 && std::string(argv[1]) == \"--replace\") {\n"
        "        ExclusiveOutput output(argv[2]);\n"
        "        if (!output.valid()) return 5;\n"
        "        if (std::remove(argv[2]) != 0) return 6;\n"
        "        std::ofstream replacement(argv[2], std::ios::binary);\n"
        "        replacement << \"foreign-replacement-must-survive\";\n"
        "        if (!replacement) return 7;\n"
        "        replacement.close();\n"
        "        return 0;\n"
        "    }\n"
        "    if (argc != 2) return 2;\n"
        "    ExclusiveOutput output(argv[1]);\n"
        "    return output.valid() ? 3 : 0;\n"
        "}\n",
        encoding="utf-8",
    )
    compiler = os.environ.get("CXX", "c++")
    rpath = f"-Wl,-rpath,{runtime}"
    command = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(source / "include"),
        "-I",
        str(source / "ggml" / "include"),
        str(harness_source),
        "-L",
        str(runtime),
        rpath,
        "-lllama",
        "-lggml",
    ]
    if sys.platform.startswith("linux"):
        command.extend(("-ldl", "-pthread"))
    command.extend(("-o", str(harness)))
    subprocess.run(command, check=True)

    inventory = subprocess.run(
        [str(harness), "--inventory"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    expected_method = (
        "macos-dyld-images" if sys.platform == "darwin" else "linux-proc-self-maps"
    )
    assert inventory[0] == expected_method
    runtime_modules = [
        Path(path).resolve(strict=True)
        for path in inventory[1:]
        if Path(path).name.lower().startswith(("libllama", "libggml"))
    ]
    assert any(path.name.lower().startswith("libllama") for path in runtime_modules)
    assert any(path.name.lower().startswith("libggml") for path in runtime_modules)
    assert all(path.parent == runtime for path in runtime_modules)

    preexisting = tmp_path / "preexisting.bin"
    original = b"must-survive-exclusive-open-failure"
    preexisting.write_bytes(original)
    completed = subprocess.run([str(harness), str(preexisting)], check=False)

    assert completed.returncode == 0
    assert preexisting.read_bytes() == original

    abandoned = tmp_path / "abandoned.bin"
    completed = subprocess.run([str(harness), str(abandoned)], check=False)
    assert completed.returncode == 3
    assert abandoned.read_bytes() == b""

    if os.name != "nt":
        replacement = tmp_path / "replacement.bin"
        completed = subprocess.run(
            [str(harness), "--replace", str(replacement)],
            check=False,
        )
        assert completed.returncode == 0
        assert replacement.read_bytes() == b"foreign-replacement-must-survive"
