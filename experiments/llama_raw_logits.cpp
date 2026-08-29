#include "llama.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: llama_raw_logits MODEL.gguf TOKENS.txt OUTPUT.bin [TOKEN_IDS]\n";
        return 2;
    }

    std::vector<llama_token> selected_tokens;
    if (argc == 5) {
        std::string value(argv[4]);
        for (char & character : value) {
            if (character == ',') {
                character = ' ';
            }
        }
        std::istringstream stream(value);
        int64_t token = 0;
        while (stream >> token) {
            if (token < 0 || token > INT32_MAX) {
                std::cerr << "invalid selected token id: " << token << "\n";
                return 2;
            }
            selected_tokens.push_back(static_cast<llama_token>(token));
        }
        if (selected_tokens.empty()) {
            std::cerr << "TOKEN_IDS contains no token ids\n";
            return 2;
        }
    }

    std::ifstream token_file(argv[2]);
    if (!token_file) {
        std::cerr << "cannot open token input: " << argv[2] << "\n";
        return 2;
    }
    std::vector<std::vector<llama_token>> rows;
    std::string line;
    while (std::getline(token_file, line)) {
        std::istringstream stream(line);
        std::vector<llama_token> tokens;
        int64_t token = 0;
        while (stream >> token) {
            if (token < 0 || token > INT32_MAX) {
                std::cerr << "invalid token id: " << token << "\n";
                return 2;
            }
            tokens.push_back(static_cast<llama_token>(token));
        }
        if (!tokens.empty()) {
            rows.push_back(std::move(tokens));
        }
    }
    if (rows.empty()) {
        std::cerr << "token input contains no rows\n";
        return 2;
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 1024;
    context_params.n_ubatch = 1024;
    context_params.n_seq_max = 1;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.embeddings = false;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token token : selected_tokens) {
        if (token >= n_vocab) {
            std::cerr << "selected token id is outside the vocabulary: " << token << "\n";
            llama_free(context);
            llama_model_free(model);
            return 2;
        }
    }
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "cannot open output: " << argv[3] << "\n";
        llama_free(context);
        llama_model_free(model);
        return 2;
    }

    for (size_t index = 0; index < rows.size(); ++index) {
        llama_memory_clear(llama_get_memory(context), true);
        llama_batch batch = llama_batch_get_one(
            rows[index].data(), static_cast<int32_t>(rows[index].size()));
        const int32_t status = llama_decode(context, batch);
        if (status != 0) {
            std::cerr << "decode failed for row " << index << ": " << status << "\n";
            llama_free(context);
            llama_model_free(model);
            return 1;
        }
        float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) {
            std::cerr << "missing last-token logits for row " << index << "\n";
            llama_free(context);
            llama_model_free(model);
            return 1;
        }
        if (selected_tokens.empty()) {
            output.write(
                reinterpret_cast<const char *>(logits),
                static_cast<std::streamsize>(n_vocab) * sizeof(float));
        } else {
            for (llama_token token : selected_tokens) {
                const float value = logits[token];
                output.write(
                    reinterpret_cast<const char *>(&value), sizeof(value));
            }
        }
        if (!output) {
            std::cerr << "output write failed at row " << index << "\n";
            llama_free(context);
            llama_model_free(model);
            return 1;
        }
        std::cerr << "{\"raw_logits_completed\":" << (index + 1)
                  << ",\"total\":" << rows.size()
                  << ",\"values_per_row\":"
                  << (selected_tokens.empty() ? n_vocab : selected_tokens.size())
                  << "}\n";
    }

    output.close();
    llama_free(context);
    llama_model_free(model);
    return 0;
}
