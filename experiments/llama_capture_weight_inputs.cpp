#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CaptureState {
    std::vector<std::string> weights;
    std::vector<std::vector<float>> values;
    int64_t row = -1;
    bool all_tokens = false;
};

bool read_rows(
    const std::string & path,
    std::vector<std::vector<llama_token>> & rows,
    size_t & maximum_tokens) {
    std::ifstream stream(path);
    if (!stream) return false;
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream input(line);
        std::vector<llama_token> tokens;
        int64_t token = 0;
        while (input >> token) {
            if (token < 0 || token > std::numeric_limits<int32_t>::max()) return false;
            tokens.push_back(static_cast<llama_token>(token));
        }
        if (!input.eof() || tokens.empty()) return false;
        maximum_tokens = std::max(maximum_tokens, tokens.size());
        rows.push_back(std::move(tokens));
    }
    return stream.eof() && !rows.empty();
}

std::vector<std::string> split_weights(const std::string & raw) {
    std::vector<std::string> result;
    std::istringstream stream(raw);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (!value.empty()) result.push_back(value);
    }
    return result;
}

int target_index(const CaptureState & state, const ggml_tensor * tensor) {
    if (tensor == nullptr) return -1;
    for (size_t index = 0; index < state.weights.size(); ++index) {
        if (state.weights[index] == tensor->name) return static_cast<int>(index);
    }
    return -1;
}

float scalar_at(const uint8_t * data, enum ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float *>(data + offset);
    }
    if (type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(
            *reinterpret_cast<const ggml_fp16_t *>(data + offset));
    }
    if (type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(
            *reinterpret_cast<const ggml_bf16_t *>(data + offset));
    }
    throw std::runtime_error(
        std::string("unsupported activation type: ") + ggml_type_name(type));
}

std::vector<float> token_vectors(const ggml_tensor * tensor, bool all_tokens) {
    if (tensor == nullptr || tensor->ne[0] < 1 || tensor->ne[1] < 1) {
        throw std::runtime_error("invalid activation tensor");
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 &&
        tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error("activation tensor is not a supported float type");
    }
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        throw std::runtime_error("expected two-dimensional projection input");
    }
    const size_t width = static_cast<size_t>(tensor->ne[0]);
    const size_t count = all_tokens ? static_cast<size_t>(tensor->ne[1]) : 1;
    const size_t first = all_tokens ? 0 : static_cast<size_t>(tensor->ne[1] - 1);
    std::vector<float> result(width * count);
    for (size_t token = 0; token < count; ++token) {
        const size_t base = (first + token) * tensor->nb[1];
        for (size_t index = 0; index < width; ++index) {
            result[token * width + index] = scalar_at(
                raw.data(), tensor->type, base + index * tensor->nb[0]);
        }
    }
    return result;
}

bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<CaptureState *>(user_data);
    if (state.row < 0 || tensor == nullptr) {
        return false;
    }
    // @name captures an explicit graph activation (e.g. @l_out-15), while
    // ordinary names retain the original projection-input capture semantics.
    int activation_index = -1;
    for (size_t i = 0; i < state.weights.size(); ++i) {
        if (state.weights[i].size() > 1 && state.weights[i][0] == '@' &&
            state.weights[i].substr(1) == tensor->name) {
            activation_index = static_cast<int>(i);
            break;
        }
    }
    if (activation_index < 0 && tensor->op != GGML_OP_MUL_MAT) return false;
    const int src0_index = target_index(state, tensor->src[0]);
    const int src1_index = target_index(state, tensor->src[1]);
    const int index = activation_index >= 0 ? activation_index :
        (src0_index >= 0 ? src0_index : src1_index);
    if (index < 0) return false;
    if (ask) return true;
    const ggml_tensor * activation = activation_index >= 0 ? tensor :
        (src0_index >= 0 ? tensor->src[1] : tensor->src[0]);
    try {
        auto captured = token_vectors(activation, state.all_tokens);
        auto & destination = state.values[static_cast<size_t>(index)];
        if (state.all_tokens) {
            destination.insert(destination.end(), captured.begin(), captured.end());
        } else {
            destination = std::move(captured);
        }
    } catch (const std::exception & error) {
        std::cerr << "capture failed for " << state.weights[static_cast<size_t>(index)]
                  << ": " << error.what() << "\n";
        return false;
    }
    return true;
}

bool write_vector(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    stream.flush();
    return stream.good();
}

void quiet_log(enum ggml_log_level, const char *, void *) {}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5 && !(argc == 6 && std::string(argv[5]) == "--all-tokens")) {
        std::cerr << "usage: llama_capture_weight_inputs MODEL.gguf TOKENS.txt "
                     "OUTPUT_DIR WEIGHT1,WEIGHT2,... [--all-tokens]\n";
        return 2;
    }
    std::vector<std::vector<llama_token>> rows;
    size_t maximum_tokens = 0;
    if (!read_rows(argv[2], rows, maximum_tokens)) {
        std::cerr << "cannot read token rows\n";
        return 2;
    }
    CaptureState state;
    state.all_tokens = argc == 6;
    state.weights = split_weights(argv[4]);
    state.values.resize(state.weights.size());
    if (state.weights.empty()) {
        std::cerr << "weight list is empty\n";
        return 2;
    }
    const std::filesystem::path output_dir(argv[3]);
    std::filesystem::create_directories(output_dir);

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::cerr << "cannot load model\n";
        llama_backend_free();
        return 1;
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(32, static_cast<uint32_t>(maximum_tokens));
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    context_params.n_seq_max = 1;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.no_perf = true;
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &state;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "cannot create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    for (size_t row = 0; row < rows.size(); ++row) {
        llama_memory_clear(llama_get_memory(context), true);
        for (auto & values : state.values) values.clear();
        state.row = static_cast<int64_t>(row);
        llama_batch batch = llama_batch_get_one(
            rows[row].data(), static_cast<int32_t>(rows[row].size()));
        if (llama_decode(context, batch) != 0) {
            std::cerr << "decode failed for row " << row << "\n";
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        for (size_t site = 0; site < state.values.size(); ++site) {
            if (state.values[site].empty()) {
                std::cerr << "weight input was not captured: " << state.weights[site] << "\n";
                llama_free(context);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            const auto path = output_dir /
                ("row" + std::to_string(row) + ".site" + std::to_string(site) + ".f32");
            if (!write_vector(path, state.values[site])) {
                std::cerr << "cannot write " << path << "\n";
                llama_free(context);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }
        const float * logits = llama_get_logits_ith(context, -1);
        const auto vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (logits == nullptr || !write_vector(
                output_dir / ("row" + std::to_string(row) + ".logits.f32"),
                std::vector<float>(logits, logits + vocab_size))) {
            std::cerr << "cannot record capture-control logits\n";
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::cerr << "{\"captured_row\":" << row
                  << ",\"site_count\":" << state.values.size() << "}\n";
    }
    state.row = -1;
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "{\"rows\":" << rows.size()
              << ",\"site_count\":" << state.weights.size() << "}\n";
    return 0;
}
