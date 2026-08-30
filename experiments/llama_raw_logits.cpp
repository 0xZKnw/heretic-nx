#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <tlhelp32.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

constexpr const char * kSchema = "llama-raw-logits-native-v2";

struct Options {
    std::string model_path;
    std::string token_path;
    std::string output_path;
    std::string backend_dir;
    std::vector<llama_token> selected_tokens;
    uint32_t n_ctx = 0;
    uint32_t n_batch = 0;
    uint32_t n_ubatch = 0;
    int32_t n_threads = 4;
    int32_t n_gpu_layers = -1;
    int32_t expected_count = 0;
    int32_t expected_vocab = 0;
    bool verbose = false;
};

class ExclusiveOutput {
public:
    explicit ExclusiveOutput(const std::string & path) {
#ifdef _WIN32
        const int descriptor = _open(
            path.c_str(), _O_BINARY | _O_CREAT | _O_EXCL | _O_WRONLY,
            _S_IREAD | _S_IWRITE);
        if (descriptor >= 0) {
            stream_ = _fdopen(descriptor, "wb");
            if (stream_ == nullptr) {
                _close(descriptor);
            }
        }
#else
        const int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (descriptor >= 0) {
            stream_ = fdopen(descriptor, "wb");
            if (stream_ == nullptr) {
                close(descriptor);
            }
        }
#endif
    }

    ExclusiveOutput(const ExclusiveOutput &) = delete;
    ExclusiveOutput & operator=(const ExclusiveOutput &) = delete;

    ~ExclusiveOutput() {
        if (stream_ != nullptr) {
            std::fclose(stream_);
        }
    }

    bool valid() const { return stream_ != nullptr; }

    bool write(const float * values, size_t count) {
        return std::fwrite(values, sizeof(float), count, stream_) == count;
    }

    bool write(float value) { return write(&value, 1); }

    bool commit() {
        if (stream_ == nullptr || std::fflush(stream_) != 0) {
            return false;
        }
#ifdef _WIN32
        if (_commit(_fileno(stream_)) != 0) {
            return false;
        }
#else
        if (fsync(fileno(stream_)) != 0) {
            return false;
        }
#endif
        if (std::fclose(stream_) != 0) {
            stream_ = nullptr;
            return false;
        }
        stream_ = nullptr;
        return true;
    }

private:
    std::FILE * stream_ = nullptr;
};

void quiet_log(enum ggml_log_level, const char *, void *) {}

std::string json_escape(const char * value) {
    std::ostringstream output;
    for (const unsigned char character : std::string(value == nullptr ? "" : value)) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    const char digits[] = "0123456789abcdef";
                    output << "\\u00" << digits[character >> 4] << digits[character & 0x0f];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string json_string_array(const std::vector<std::string> & values) {
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << '"' << json_escape(values[index].c_str()) << '"';
    }
    output << ']';
    return output.str();
}

#ifdef _WIN32
bool wide_to_utf8(const wchar_t * value, std::string & result) {
    const int size = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr, 0, nullptr, nullptr);
    if (size < 1) {
        return false;
    }
    std::vector<char> buffer(static_cast<size_t>(size));
    if (WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, buffer.data(), size,
            nullptr, nullptr) != size) {
        return false;
    }
    result.assign(buffer.data());
    return true;
}
#elif defined(__linux__)
std::string decode_proc_maps_path(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '\\' && index + 3 < value.size() &&
            value[index + 1] >= '0' && value[index + 1] <= '7' &&
            value[index + 2] >= '0' && value[index + 2] <= '7' &&
            value[index + 3] >= '0' && value[index + 3] <= '7') {
            const int decoded =
                (value[index + 1] - '0') * 64 +
                (value[index + 2] - '0') * 8 +
                (value[index + 3] - '0');
            result.push_back(static_cast<char>(decoded));
            index += 3;
        } else {
            result.push_back(value[index]);
        }
    }
    return result;
}

bool linux_shared_object_path(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    const std::string name = slash == std::string::npos
        ? path
        : path.substr(slash + 1);
    return name.find(".so") != std::string::npos;
}
#endif

bool collect_loaded_modules(
    std::vector<std::string> & modules, std::string & inventory_method) {
    std::set<std::string> unique;
#ifdef _WIN32
    const HANDLE snapshot = CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
        return false;
    }
    MODULEENTRY32W entry = {};
    entry.dwSize = sizeof(entry);
    if (!Module32FirstW(snapshot, &entry)) {
        CloseHandle(snapshot);
        return false;
    }
    do {
        std::string path;
        if (!wide_to_utf8(entry.szExePath, path) || path.empty()) {
            CloseHandle(snapshot);
            return false;
        }
        unique.insert(path);
    } while (Module32NextW(snapshot, &entry));
    const DWORD final_error = GetLastError();
    CloseHandle(snapshot);
    if (final_error != ERROR_NO_MORE_FILES) {
        return false;
    }
    inventory_method = "windows-toolhelp32";
#elif defined(__APPLE__)
    const uint32_t count = _dyld_image_count();
    for (uint32_t index = 0; index < count; ++index) {
        const char * path = _dyld_get_image_name(index);
        if (path == nullptr || path[0] == '\0') {
            return false;
        }
        unique.insert(path);
    }
    inventory_method = "macos-dyld-images";
#elif defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    if (!maps) {
        return false;
    }
    std::string line;
    while (std::getline(maps, line)) {
        const size_t path_start = line.find('/');
        if (path_start == std::string::npos) {
            continue;
        }
        const std::string path = decode_proc_maps_path(line.substr(path_start));
        if (path.find(" (deleted)") != std::string::npos) {
            return false;
        }
        if (linux_shared_object_path(path)) {
            unique.insert(path);
        }
    }
    if (!maps.eof()) {
        return false;
    }
    char executable_path[4096] = {};
    const ssize_t executable_size = readlink(
        "/proc/self/exe", executable_path, sizeof(executable_path) - 1);
    if (executable_size < 1 ||
        executable_size >= static_cast<ssize_t>(sizeof(executable_path))) {
        return false;
    }
    executable_path[executable_size] = '\0';
    unique.insert(executable_path);
    inventory_method = "linux-proc-self-maps";
#else
    (void) unique;
    return false;
#endif
    if (unique.empty()) {
        return false;
    }
    modules.assign(unique.begin(), unique.end());
    return true;
}

bool host_is_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

bool parse_integer(const std::string & value, int64_t & result) {
    try {
        size_t consumed = 0;
        result = std::stoll(value, &consumed, 10);
        return consumed == value.size();
    } catch (const std::exception &) {
        return false;
    }
}

bool parse_positive_u32(const std::string & value, uint32_t & result) {
    int64_t parsed = 0;
    if (!parse_integer(value, parsed) || parsed < 1 ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_positive_i32(const std::string & value, int32_t & result) {
    int64_t parsed = 0;
    if (!parse_integer(value, parsed) || parsed < 1 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    result = static_cast<int32_t>(parsed);
    return true;
}

bool parse_selected_tokens(
    const std::string & value, std::vector<llama_token> & selected_tokens) {
    std::string normalized(value);
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    std::set<llama_token> seen;
    int64_t token = 0;
    while (stream >> token) {
        if (token < 0 || token > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        const auto converted = static_cast<llama_token>(token);
        if (!seen.insert(converted).second) {
            return false;
        }
        selected_tokens.push_back(converted);
    }
    return stream.eof() && !selected_tokens.empty();
}

void usage() {
    std::cerr
        << "usage: llama_raw_logits MODEL.gguf TOKENS.txt OUTPUT.bin [TOKEN_IDS] "
        << "[--n-ctx N] [--n-batch N] [--n-ubatch N] [--threads N] "
        << "[--gpu-layers N] [--expected-count N] [--expected-vocab N] "
        << "[--backend-dir DIR] [--verbose]\n";
}

bool parse_options(int argc, char ** argv, Options & options) {
    if (argc < 4) {
        usage();
        return false;
    }
    options.model_path = argv[1];
    options.token_path = argv[2];
    options.output_path = argv[3];
    int index = 4;
    if (index < argc && std::string(argv[index]).rfind("--", 0) != 0) {
        if (!parse_selected_tokens(argv[index], options.selected_tokens)) {
            std::cerr << "TOKEN_IDS is empty, malformed, or contains duplicates\n";
            return false;
        }
        ++index;
    }
    while (index < argc) {
        const std::string flag(argv[index++]);
        if (flag == "--verbose") {
            options.verbose = true;
            continue;
        }
        if (index >= argc) {
            std::cerr << "missing value for " << flag << "\n";
            return false;
        }
        const std::string value(argv[index++]);
        if (flag == "--n-ctx") {
            if (!parse_positive_u32(value, options.n_ctx)) return false;
        } else if (flag == "--n-batch") {
            if (!parse_positive_u32(value, options.n_batch)) return false;
        } else if (flag == "--n-ubatch") {
            if (!parse_positive_u32(value, options.n_ubatch)) return false;
        } else if (flag == "--threads") {
            if (!parse_positive_i32(value, options.n_threads)) return false;
        } else if (flag == "--gpu-layers") {
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) ||
                parsed < std::numeric_limits<int32_t>::min() ||
                parsed > std::numeric_limits<int32_t>::max()) return false;
            options.n_gpu_layers = static_cast<int32_t>(parsed);
        } else if (flag == "--expected-count") {
            if (!parse_positive_i32(value, options.expected_count)) return false;
        } else if (flag == "--expected-vocab") {
            if (!parse_positive_i32(value, options.expected_vocab)) return false;
        } else if (flag == "--backend-dir") {
            if (value.empty()) return false;
            options.backend_dir = value;
        } else {
            std::cerr << "unknown option: " << flag << "\n";
            return false;
        }
    }
    return true;
}

bool read_token_rows(
    const std::string & path,
    std::vector<std::vector<llama_token>> & rows,
    size_t & maximum_tokens) {
    std::ifstream token_file(path);
    if (!token_file) {
        std::cerr << "cannot open token input: " << path << "\n";
        return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(token_file, line)) {
        ++line_number;
        std::istringstream stream(line);
        std::vector<llama_token> tokens;
        int64_t token = 0;
        while (stream >> token) {
            if (token < 0 || token > std::numeric_limits<int32_t>::max()) {
                std::cerr << "invalid token id on input line " << line_number << "\n";
                return false;
            }
            tokens.push_back(static_cast<llama_token>(token));
        }
        if (!stream.eof() || tokens.empty()) {
            std::cerr << "empty or malformed token input line " << line_number << "\n";
            return false;
        }
        maximum_tokens = std::max(maximum_tokens, tokens.size());
        rows.push_back(std::move(tokens));
    }
    if (!token_file.eof()) {
        std::cerr << "failed while reading token input: " << path << "\n";
        return false;
    }
    if (rows.empty()) {
        std::cerr << "token input contains no rows\n";
        return false;
    }
    return true;
}

double seconds_since(const std::chrono::steady_clock::time_point & started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

}  // namespace

int main(int argc, char ** argv) {
    const auto total_started = std::chrono::steady_clock::now();
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    if (!host_is_little_endian() || sizeof(float) != 4) {
        std::cerr << "raw-logit format requires little-endian float32\n";
        return 2;
    }

    std::vector<std::vector<llama_token>> rows;
    size_t maximum_tokens = 0;
    if (!read_token_rows(options.token_path, rows, maximum_tokens)) {
        return 2;
    }
    if (options.expected_count > 0 &&
        rows.size() != static_cast<size_t>(options.expected_count)) {
        std::cerr << "token row count does not match --expected-count\n";
        return 2;
    }
    if (maximum_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        std::cerr << "prompt is too large for llama.cpp context parameters\n";
        return 2;
    }

    const uint32_t requested_ctx = options.n_ctx == 0
        ? std::max<uint32_t>(32, static_cast<uint32_t>(maximum_tokens))
        : options.n_ctx;
    const uint32_t requested_batch = options.n_batch == 0
        ? std::max<uint32_t>(32, static_cast<uint32_t>(maximum_tokens))
        : options.n_batch;
    const uint32_t requested_ubatch = options.n_ubatch == 0
        ? requested_batch
        : options.n_ubatch;
    if (maximum_tokens > requested_ctx || maximum_tokens > requested_batch ||
        requested_ubatch > requested_batch) {
        std::cerr << "maximum prompt length exceeds configured context or batch\n";
        return 2;
    }

    if (!options.verbose) {
        llama_log_set(quiet_log, nullptr);
    }
    if (options.backend_dir.empty()) {
        ggml_backend_load_all();
    } else {
        ggml_backend_load_all_from_path(options.backend_dir.c_str());
    }
    llama_backend_init();
    const bool gpu_offload_supported = llama_supports_gpu_offload();
    if (options.n_gpu_layers != 0 && !gpu_offload_supported) {
        std::cerr << "GPU layer offload was requested but the loaded backend "
                     "does not support it; pass --gpu-layers 0 for an explicit "
                     "CPU run\n";
        llama_backend_free();
        return 2;
    }
    const auto load_started = std::chrono::steady_clock::now();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(
        options.model_path.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "failed to load model\n";
        llama_backend_free();
        return 1;
    }
    const double model_load_seconds = seconds_since(load_started);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab < 2 ||
        (options.expected_vocab > 0 && n_vocab != options.expected_vocab)) {
        std::cerr << "model vocabulary does not match --expected-vocab\n";
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    for (const auto & row : rows) {
        for (const llama_token token : row) {
            if (token >= n_vocab) {
                std::cerr << "prompt token id is outside the model vocabulary: "
                          << token << "\n";
                llama_model_free(model);
                llama_backend_free();
                return 2;
            }
        }
    }
    for (const llama_token token : options.selected_tokens) {
        if (token >= n_vocab) {
            std::cerr << "selected token id is outside the vocabulary: " << token << "\n";
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
    }
    const int32_t model_context_train = llama_model_n_ctx_train(model);
    if (model_context_train < 1 ||
        maximum_tokens > static_cast<size_t>(model_context_train)) {
        std::cerr << "maximum prompt length exceeds the model training context\n";
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = requested_ctx;
    context_params.n_batch = requested_batch;
    context_params.n_ubatch = requested_ubatch;
    context_params.n_seq_max = 1;
    context_params.n_threads = options.n_threads;
    context_params.n_threads_batch = options.n_threads;
    context_params.embeddings = false;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (maximum_tokens > llama_n_ctx_seq(context) ||
        maximum_tokens > llama_n_batch(context)) {
        std::cerr << "actual llama.cpp context cannot fit the longest prompt\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ExclusiveOutput output(options.output_path);
    if (!output.valid()) {
        std::cerr << "cannot create output exclusively: " << options.output_path
                  << ": " << std::strerror(errno) << "\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    const auto decode_started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < rows.size(); ++index) {
        llama_memory_clear(llama_get_memory(context), true);
        llama_batch batch = llama_batch_get_one(
            rows[index].data(), static_cast<int32_t>(rows[index].size()));
        const int32_t status = llama_decode(context, batch);
        if (status != 0) {
            std::cerr << "decode failed for row " << index << ": " << status << "\n";
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) {
            std::cerr << "missing last-token logits for row " << index << "\n";
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        bool wrote = true;
        if (options.selected_tokens.empty()) {
            wrote = output.write(logits, static_cast<size_t>(n_vocab));
        } else {
            for (const llama_token token : options.selected_tokens) {
                wrote = output.write(logits[token]);
                if (!wrote) break;
            }
        }
        if (!wrote) {
            std::cerr << "output write failed at row " << index << "\n";
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::cerr << "{\"raw_logits_completed\":" << (index + 1)
                  << ",\"total\":" << rows.size()
                  << ",\"values_per_row\":"
                  << (options.selected_tokens.empty()
                          ? static_cast<size_t>(n_vocab)
                          : options.selected_tokens.size())
                  << "}\n";
    }
    const double decode_seconds = seconds_since(decode_started);
    std::vector<std::string> loaded_modules;
    std::string module_inventory_method;
    if (!collect_loaded_modules(loaded_modules, module_inventory_method)) {
        std::cerr << "cannot attest the loaded native runtime modules\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (!output.commit()) {
        std::cerr << "failed to flush and sync raw-logit output\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    char model_description[512] = {};
    llama_model_desc(model, model_description, sizeof(model_description));
    const std::string system_info = llama_print_system_info();
    const uint32_t actual_ctx = llama_n_ctx(context);
    const uint32_t actual_batch = llama_n_batch(context);
    const uint32_t actual_ubatch = llama_n_ubatch(context);
    const uint64_t model_size = llama_model_size(model);
    const uint64_t parameter_count = llama_model_n_params(model);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();

    std::cout
        << "{\"schema_version\":\"" << kSchema << "\""
        << ",\"count\":" << rows.size()
        << ",\"vocab_size\":" << n_vocab
        << ",\"values_per_row\":"
        << (options.selected_tokens.empty()
                ? static_cast<size_t>(n_vocab)
                : options.selected_tokens.size())
        << ",\"maximum_prompt_tokens\":" << maximum_tokens
        << ",\"n_ctx\":" << actual_ctx
        << ",\"n_batch\":" << actual_batch
        << ",\"n_ubatch\":" << actual_ubatch
        << ",\"threads\":" << options.n_threads
        << ",\"gpu_layers\":" << options.n_gpu_layers
        << ",\"gpu_offload_supported\":"
        << (gpu_offload_supported ? "true" : "false")
        << ",\"model_context_train\":" << model_context_train
        << ",\"model_size_bytes\":" << model_size
        << ",\"model_parameter_count\":" << parameter_count
        << ",\"model_description\":\"" << json_escape(model_description) << "\""
        << ",\"system_info\":\"" << json_escape(system_info.c_str()) << "\""
        << ",\"module_inventory_method\":\""
        << json_escape(module_inventory_method.c_str()) << "\""
        << ",\"loaded_modules\":" << json_string_array(loaded_modules)
        << ",\"float_format\":\"float32-little-endian\""
        << ",\"memory_cleared_between_rows\":true"
        << ",\"backend_loading\":\""
        << (options.backend_dir.empty() ? "default-search" : "explicit-directory")
        << "\""
        << ",\"model_load_seconds\":" << model_load_seconds
        << ",\"decode_seconds\":" << decode_seconds
        << ",\"total_seconds\":" << seconds_since(total_started)
        << "}\n";
    return 0;
}
