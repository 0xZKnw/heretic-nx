#define main capture_collector_main
#include "../experiments/llama_capture_weight_inputs.cpp"
#undef main
#include <cassert>

int main() {
    CaptureState state;
    state.row = 0;
    state.weights = {"@l_out-15", "blk.1.ffn_down.weight"};
    state.values.resize(2);
    ggml_tensor residual{};
    ggml_set_name(&residual, "l_out-15");
    residual.op = GGML_OP_ADD;
    assert(capture_callback(&residual, true, &state));
    ggml_set_name(&residual, "l_out-14");
    assert(!capture_callback(&residual, true, &state));
    ggml_tensor weight{}, input{}, projection{};
    ggml_set_name(&weight, "blk.1.ffn_down.weight");
    projection.op = GGML_OP_MUL_MAT;
    projection.src[0] = &weight;
    projection.src[1] = &input;
    assert(capture_callback(&projection, true, &state));
    projection.op = GGML_OP_ADD;
    assert(!capture_callback(&projection, true, &state));
    state.row = -1;
    ggml_set_name(&residual, "l_out-15");
    assert(!capture_callback(&residual, true, &state));
    assert(!capture_callback(nullptr, true, &state));
    return 0;
}
