#!/bin/sh
set -eu

llama_cpp_revision=d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
llama_cpp_source=${1:-"$project_root/references/llama.cpp"}
native_build_dir=${2:-"$project_root/build/llama.cpp-native"}
cpp_compiler=${CXX:-c++}
cmake_command=${CMAKE:-cmake}

if [ ! -d "$llama_cpp_source/.git" ]; then
    echo "missing llama.cpp git checkout: $llama_cpp_source" >&2
    echo "checkout https://github.com/ggml-org/llama.cpp at $llama_cpp_revision" >&2
    exit 2
fi

actual_revision=$(git -C "$llama_cpp_source" rev-parse HEAD)
if [ "$actual_revision" != "$llama_cpp_revision" ]; then
    echo "llama.cpp revision mismatch: $actual_revision != $llama_cpp_revision" >&2
    exit 2
fi
if ! git -C "$llama_cpp_source" diff --quiet --ignore-submodules --; then
    echo "llama.cpp checkout has tracked modifications; refusing an unpinned build" >&2
    exit 2
fi
if ! git -C "$llama_cpp_source" diff --cached --quiet --ignore-submodules --; then
    echo "llama.cpp checkout has staged modifications; refusing an unpinned build" >&2
    exit 2
fi

case $(uname -s) in
    Darwin)
        cmake_runtime_rpath='@loader_path'
        runtime_rpath='-Wl,-rpath,@loader_path'
        platform_libraries=''
        ggml_metal=ON
        ggml_metal_embed_library=ON
        ;;
    Linux)
        cmake_runtime_rpath='$ORIGIN'
        runtime_rpath='-Wl,-rpath,$ORIGIN'
        platform_libraries='-ldl -pthread'
        ggml_metal=OFF
        ggml_metal_embed_library=OFF
        ;;
    *)
        echo "native raw-logit build recipe currently supports macOS and Linux" >&2
        exit 2
        ;;
esac

if [ -f "$native_build_dir/CMakeCache.txt" ]; then
    "$cmake_command" --build "$native_build_dir" --target clean
fi

"$cmake_command" \
    -S "$llama_cpp_source" \
    -B "$native_build_dir" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_METAL="$ggml_metal" \
    -DGGML_METAL_EMBED_LIBRARY="$ggml_metal_embed_library" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DCMAKE_BUILD_RPATH="$cmake_runtime_rpath" \
    -DCMAKE_INSTALL_RPATH="$cmake_runtime_rpath" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_BUILD_TYPE=Release
"$cmake_command" --build "$native_build_dir" --target llama --parallel

runtime_dir="$native_build_dir/bin"
mkdir -p "$runtime_dir"
temporary_output=$(mktemp "$runtime_dir/.llama_raw_logits.XXXXXX")
trap 'rm -f "$temporary_output"' EXIT HUP INT TERM

# platform_libraries deliberately expands to two fixed linker flags on Linux.
# shellcheck disable=SC2086
"$cpp_compiler" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -Wall \
    -Wextra \
    -Werror \
    -I "$llama_cpp_source/include" \
    -I "$llama_cpp_source/ggml/include" \
    "$project_root/experiments/llama_raw_logits.cpp" \
    -L "$runtime_dir" \
    "$runtime_rpath" \
    -lllama \
    -lggml \
    $platform_libraries \
    -o "$temporary_output"
chmod 0755 "$temporary_output"
mv -f "$temporary_output" "$runtime_dir/llama_raw_logits"
trap - EXIT HUP INT TERM

echo "$runtime_dir/llama_raw_logits"
