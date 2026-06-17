// Fairy2i CPU backend smoke tests.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

extern "C" {
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool test_fairy2i_wide_linear_w2_zero() {
    constexpr int64_t K = QK_FAIRY2I_TILE64;
    constexpr int64_t M = 4;
    constexpr int64_t N = 3;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);

    struct ggml_init_params params = {
        /*.mem_size   =*/16 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * u_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, K, M);
    ggml_tensor * u_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, K, M);
    ggml_tensor * w_s0 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, K, M);
    ggml_tensor * w_s1 = ggml_new_tensor_2d(ctx, GGML_TYPE_FAIRY2I_TILE64_V2, K, M);
    ggml_tensor * out  = ggml_fairy2i_wide_linear_w2(ctx, x, u_s0, u_s1, w_s0, w_s1, NULL);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate backend buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const size_t blocks = (size_t) M * (size_t) K / QK_FAIRY2I_TILE64;
    std::vector<float>                   x_data((size_t) N * (size_t) K, 0.0f);
    std::vector<block_fairy2i_tile64_v2> weights(blocks);

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(u_s0, weights.data(), 0, weights.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(u_s1, weights.data(), 0, weights.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s0, weights.data(), 0, weights.size() * sizeof(block_fairy2i_tile64_v2));
    ggml_backend_tensor_set(w_s1, weights.data(), 0, weights.size() * sizeof(block_fairy2i_tile64_v2));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Fairy2i W2 graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> out_data((size_t) M * (size_t) N, -1.0f);
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

    bool ok = true;
    for (float v : out_data) {
        if (v != 0.0f) {
            ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  fairy2i wide_linear_w2 zero - %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_cpu_init();

    printf("========================================\n");
    printf("Fairy2i CPU Unit Tests\n");
    printf("========================================\n");

    int num_failed = 0;
    if (!test_fairy2i_wide_linear_w2_zero()) {
        fprintf(stderr, "Fairy2i W2 zero test FAILED\n");
        ++num_failed;
    }

    printf("\n========================================\n");
    if (num_failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", num_failed);
    }
    printf("========================================\n");

    return num_failed == 0 ? 0 : 1;
}
