"""Same-type GGUF quantization codecs with an optional native ggml backend."""

from __future__ import annotations

import ctypes
import ctypes.util
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import threading
from typing import Any

import numpy as np

from heretic_nx.hashing import sha256_file


@dataclass(frozen=True)
class QuantLayout:
    block_size: int
    type_size: int
    native_suffix: str
    requires_native: bool = False


QUANT_LAYOUTS: dict[str, QuantLayout] = {
    "Q4_0": QuantLayout(32, 18, "q4_0"),
    "Q4_1": QuantLayout(32, 20, "q4_1"),
    "Q5_0": QuantLayout(32, 22, "q5_0"),
    "Q5_1": QuantLayout(32, 24, "q5_1"),
    "Q8_0": QuantLayout(32, 34, "q8_0"),
    "Q2_K": QuantLayout(256, 84, "q2_K", requires_native=True),
    "Q3_K": QuantLayout(256, 110, "q3_K", requires_native=True),
    "Q4_K": QuantLayout(256, 144, "q4_K", requires_native=True),
    "Q5_K": QuantLayout(256, 176, "q5_K", requires_native=True),
    "Q6_K": QuantLayout(256, 210, "q6_K", requires_native=True),
}


DEFAULT_PARALLEL_MIN_ELEMENTS = 65_536
MAX_AUTO_QUANTIZATION_THREADS = 8


def _available_cpu_count() -> int:
    """Return an affinity-aware CPU count when the platform exposes one."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return max(1, len(get_affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _resolve_quantization_threads(value: int | None) -> int:
    """Resolve explicit/env/auto native quantization parallelism."""

    if value is None:
        configured = os.environ.get("HERETIC_NX_QUANT_THREADS")
        if configured and configured.strip().lower() != "auto":
            try:
                value = int(configured)
            except ValueError as error:
                raise ValueError(
                    "HERETIC_NX_QUANT_THREADS must be 'auto' or a positive integer"
                ) from error
    if value is None:
        return min(MAX_AUTO_QUANTIZATION_THREADS, _available_cpu_count())
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("quantization_threads must be a positive integer or None")
    return value


def _library_globs(directory: Path) -> tuple[Path, ...]:
    patterns = (
        "libggml-base*.dylib",
        "libggml-base*.so",
        "libggml-base.so.*",
        "ggml-base*.dll",
    )
    return tuple(path for pattern in patterns for path in sorted(directory.glob(pattern)))


def _native_library_candidates(explicit: str | Path | None) -> tuple[str | Path, ...]:
    if explicit is not None:
        return (Path(explicit).expanduser(),)

    candidates: list[str | Path] = []
    configured = os.environ.get("HERETIC_NX_GGML_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())

    repository = Path(__file__).resolve().parents[3]
    candidates.extend(_library_globs(repository / "references" / "llama.cpp" / "build" / "bin"))

    quantize_binary = shutil.which("llama-quantize")
    if quantize_binary:
        candidates.extend(_library_globs(Path(quantize_binary).resolve().parent))

    discovered = ctypes.util.find_library("ggml-base")
    if discovered:
        candidates.append(discovered)

    unique: list[str | Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


class NativeGGMLCodec:
    """ctypes binding to llama.cpp's reference-compatible row codecs."""

    def __init__(
        self,
        library: str | Path | None = None,
        *,
        quantization_threads: int | None = None,
        parallel_min_elements: int = DEFAULT_PARALLEL_MIN_ELEMENTS,
    ) -> None:
        if (
            isinstance(parallel_min_elements, bool)
            or not isinstance(parallel_min_elements, int)
            or parallel_min_elements < 1
        ):
            raise ValueError("parallel_min_elements must be a positive integer")
        self.quantization_threads = _resolve_quantization_threads(
            quantization_threads
        )
        self.parallel_min_elements = parallel_min_elements
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        failures: list[str] = []
        explicit = library is not None
        for candidate in _native_library_candidates(library):
            try:
                path = (
                    Path(candidate).resolve(strict=True)
                    if isinstance(candidate, Path)
                    else candidate
                )
                loaded = ctypes.CDLL(str(path))
                self._configure(loaded)
                self._validate_abi(loaded)
                self._library = loaded
                self.path = str(path)
                self.sha256 = sha256_file(path) if isinstance(path, Path) else None
                return
            except (AttributeError, OSError, RuntimeError, ValueError) as error:
                failures.append(f"{candidate}: {error}")
                if explicit:
                    break
        detail = "; ".join(failures) if failures else "no candidate library was found"
        raise RuntimeError(
            "K-quant editing requires a compatible libggml-base. Pass --ggml-library, "
            "set HERETIC_NX_GGML_LIBRARY, or install/build llama.cpp. "
            f"Discovery details: {detail}"
        )

    def _get_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is not None:
            return executor
        with self._executor_lock:
            executor = self._executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=self.quantization_threads,
                    thread_name_prefix="hnx-ggml-quant",
                )
                self._executor = executor
        return executor

    def close(self) -> None:
        """Release lazily-created native quantization workers."""

        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self) -> "NativeGGMLCodec":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _configure(library: ctypes.CDLL) -> None:
        required_symbols = (
            "ggml_blck_size",
            "ggml_type_size",
            "ggml_validate_row_data",
            *(f"quantize_{layout.native_suffix}" for layout in QUANT_LAYOUTS.values()),
            *(
                f"dequantize_row_{layout.native_suffix}"
                for layout in QUANT_LAYOUTS.values()
            ),
        )
        missing = [name for name in required_symbols if not hasattr(library, name)]
        if missing:
            raise RuntimeError(
                "libggml-base is missing required codec symbols: "
                + ", ".join(missing)
            )

        library.ggml_blck_size.argtypes = [ctypes.c_int]
        library.ggml_blck_size.restype = ctypes.c_int64
        library.ggml_type_size.argtypes = [ctypes.c_int]
        library.ggml_type_size.restype = ctypes.c_size_t
        library.ggml_validate_row_data.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.ggml_validate_row_data.restype = ctypes.c_bool

        for layout in QUANT_LAYOUTS.values():
            quantize = getattr(library, f"quantize_{layout.native_suffix}")
            quantize.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
            ]
            quantize.restype = ctypes.c_size_t
            dequantize = getattr(library, f"dequantize_row_{layout.native_suffix}")
            dequantize.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
            ]
            dequantize.restype = None

    @staticmethod
    def _validate_abi(library: ctypes.CDLL) -> None:
        try:
            from gguf import GGMLQuantizationType
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("native GGUF editing requires the 'gguf' extra") from error

        for name, layout in QUANT_LAYOUTS.items():
            qtype = int(getattr(GGMLQuantizationType, name))
            block_size = int(library.ggml_blck_size(qtype))
            type_size = int(library.ggml_type_size(qtype))
            if (block_size, type_size) != (layout.block_size, layout.type_size):
                raise RuntimeError(
                    f"libggml ABI mismatch for {name}: {(block_size, type_size)} != "
                    f"{(layout.block_size, layout.type_size)}"
                )

    def validate_payload(self, encoded: np.ndarray, qtype: Any) -> None:
        payload = np.ascontiguousarray(encoded).view(np.uint8)
        if not self._library.ggml_validate_row_data(
            int(qtype),
            payload.ctypes.data_as(ctypes.c_void_p),
            payload.nbytes,
        ):
            raise RuntimeError(f"libggml rejected a {qtype.name} row payload")

    def dequantize_rows(
        self,
        encoded: np.ndarray,
        qtype: Any,
        input_dim: int,
    ) -> np.ndarray:
        layout = QUANT_LAYOUTS[qtype.name]
        payload = np.ascontiguousarray(encoded).view(np.uint8)
        expected_row_bytes = input_dim // layout.block_size * layout.type_size
        if payload.ndim != 2 or payload.shape[1] != expected_row_bytes:
            raise ValueError(
                f"invalid {qtype.name} payload shape {payload.shape}; "
                f"expected (*, {expected_row_bytes})"
            )
        self.validate_payload(payload, qtype)
        output = np.empty((payload.shape[0], input_dim), dtype=np.float32)
        dequantize = getattr(self._library, f"dequantize_row_{layout.native_suffix}")
        dequantize(
            payload.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.size,
        )
        if not np.isfinite(output).all():
            raise RuntimeError(f"native {qtype.name} dequantization produced non-finite values")
        return output

    def quantize_rows(self, values: np.ndarray, qtype: Any) -> np.ndarray:
        matrix = np.ascontiguousarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] % QUANT_LAYOUTS[qtype.name].block_size:
            raise ValueError(f"invalid matrix shape {matrix.shape} for {qtype.name}")
        if not np.isfinite(matrix).all():
            raise ValueError("cannot quantize non-finite values")
        layout = QUANT_LAYOUTS[qtype.name]
        row_bytes = matrix.shape[1] // layout.block_size * layout.type_size
        output = np.empty((matrix.shape[0], row_bytes), dtype=np.uint8)
        quantize = getattr(self._library, f"quantize_{layout.native_suffix}")

        def quantize_range(start: int, stop: int) -> None:
            source_rows = matrix[start:stop]
            output_rows = output[start:stop]
            written = int(
                quantize(
                    source_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output_rows.ctypes.data_as(ctypes.c_void_p),
                    stop - start,
                    matrix.shape[1],
                    None,
                )
            )
            if written != output_rows.nbytes:
                raise RuntimeError(
                    f"native {qtype.name} quantizer wrote {written} bytes; "
                    f"expected {output_rows.nbytes}"
                )

        worker_count = (
            min(self.quantization_threads, matrix.shape[0])
            if self.quantization_threads > 1
            and matrix.size >= self.parallel_min_elements
            else 1
        )
        if worker_count == 1:
            quantize_range(0, matrix.shape[0])
        else:
            rows_per_worker, remainder = divmod(matrix.shape[0], worker_count)
            ranges = []
            start = 0
            for worker_index in range(worker_count):
                stop = start + rows_per_worker + int(worker_index < remainder)
                ranges.append((start, stop))
                start = stop
            executor = self._get_executor()
            futures = [
                executor.submit(quantize_range, start, stop)
                for start, stop in ranges
            ]
            for future in futures:
                future.result()
        self.validate_payload(output, qtype)
        return output

    def provenance(self) -> dict[str, object]:
        return {
            "backend": "libggml-base",
            "path": self.path,
            "sha256": self.sha256,
            "quantization_threads": self.quantization_threads,
            "parallel_min_elements": self.parallel_min_elements,
        }


class GGUFQuantizationCodecRegistry:
    """Dispatch supported tensor types to native ggml or gguf-python codecs."""

    def __init__(
        self,
        *,
        ggml_library: str | Path | None = None,
        prefer_native: bool = True,
        quantization_threads: int | None = None,
        parallel_min_elements: int = DEFAULT_PARALLEL_MIN_ELEMENTS,
    ) -> None:
        if (
            isinstance(parallel_min_elements, bool)
            or not isinstance(parallel_min_elements, int)
            or parallel_min_elements < 1
        ):
            raise ValueError("parallel_min_elements must be a positive integer")
        self._ggml_library = ggml_library
        self._prefer_native = prefer_native
        self._quantization_threads = quantization_threads
        self._parallel_min_elements = parallel_min_elements
        self._native: NativeGGMLCodec | None = None
        self._native_unavailable = False

    def _get_native(self, *, required: bool) -> NativeGGMLCodec | None:
        if self._native is not None:
            return self._native
        if self._native_unavailable and not required:
            return None
        try:
            self._native = NativeGGMLCodec(
                self._ggml_library,
                quantization_threads=self._quantization_threads,
                parallel_min_elements=self._parallel_min_elements,
            )
            return self._native
        except RuntimeError:
            if required or self._ggml_library is not None:
                raise
            self._native_unavailable = True
            return None

    def close(self) -> None:
        """Release native worker threads, if native quantization was used."""

        if self._native is not None:
            self._native.close()

    def __enter__(self) -> "GGUFQuantizationCodecRegistry":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ensure_supported(self, qtype: Any) -> None:
        layout = QUANT_LAYOUTS.get(qtype.name)
        if layout is None:
            raise RuntimeError(
                f"direct same-type editing does not support {qtype.name}; supported types are "
                f"{sorted(QUANT_LAYOUTS)}"
            )
        if layout.requires_native:
            self._get_native(required=True)
        elif self._prefer_native:
            self._get_native(required=False)

    def dequantize_rows(
        self,
        encoded: np.ndarray,
        qtype: Any,
        input_dim: int,
    ) -> np.ndarray:
        self.ensure_supported(qtype)
        requires_native = QUANT_LAYOUTS[qtype.name].requires_native
        native = (
            self._get_native(required=requires_native)
            if requires_native or self._prefer_native
            else None
        )
        if native is not None:
            return native.dequantize_rows(encoded, qtype, input_dim)
        from gguf.quants import dequantize

        layout = QUANT_LAYOUTS[qtype.name]
        payload = np.ascontiguousarray(encoded).view(np.uint8)
        expected_row_bytes = input_dim // layout.block_size * layout.type_size
        if payload.ndim != 2 or payload.shape[1] != expected_row_bytes:
            raise ValueError(
                f"invalid {qtype.name} payload shape {payload.shape}; "
                f"expected (*, {expected_row_bytes})"
            )
        output = np.ascontiguousarray(dequantize(payload, qtype), dtype=np.float32)
        if output.shape != (encoded.shape[0], input_dim):
            raise RuntimeError(
                f"gguf-python dequantized {qtype.name} to {output.shape}; expected "
                f"{(encoded.shape[0], input_dim)}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError(
                f"gguf-python {qtype.name} dequantization produced non-finite values"
            )
        return output

    def quantize_rows(self, values: np.ndarray, qtype: Any) -> np.ndarray:
        self.ensure_supported(qtype)
        requires_native = QUANT_LAYOUTS[qtype.name].requires_native
        native = (
            self._get_native(required=requires_native)
            if requires_native or self._prefer_native
            else None
        )
        if native is not None:
            return native.quantize_rows(values, qtype)
        from gguf.quants import quantize

        matrix = np.ascontiguousarray(values, dtype=np.float32)
        layout = QUANT_LAYOUTS[qtype.name]
        if matrix.ndim != 2 or matrix.shape[1] % layout.block_size:
            raise ValueError(f"invalid matrix shape {matrix.shape} for {qtype.name}")
        if not np.isfinite(matrix).all():
            raise ValueError("cannot quantize non-finite values")
        output = np.ascontiguousarray(quantize(matrix, qtype)).view(np.uint8)
        expected_shape = (
            matrix.shape[0],
            matrix.shape[1] // layout.block_size * layout.type_size,
        )
        if output.shape != expected_shape:
            raise RuntimeError(
                f"gguf-python quantized {qtype.name} to {output.shape}; "
                f"expected {expected_shape}"
            )
        return output

    def backend_for(self, qtype: Any) -> str:
        layout = QUANT_LAYOUTS.get(qtype.name)
        if layout is not None and layout.requires_native:
            return "libggml-base" if self._native is not None else "unavailable"
        if self._prefer_native and self._native is not None:
            return "libggml-base"
        return "gguf-python"

    def provenance(self) -> dict[str, object]:
        if self._native is None:
            return {"backend": "gguf-python"}
        return self._native.provenance()
