// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static bool mixed_zero_imatrix_quantization_is_deterministic(ggml_type     type,
                                                             size_t        test_size,
                                                             const float * test_data) {
    const size_t         row_size = ggml_row_size(type, test_size);
    std::vector<uint8_t> first(row_size, 0xa5);
    std::vector<uint8_t> second(row_size, 0x5a);
    std::vector<float>   imatrix(test_size, 1.0f);
    for (size_t i = 0; i < test_size; ++i) {
        if ((i / 32) % 2 == 0) {
            imatrix[i] = 0.0f;
        }
    }

    const size_t first_size  = ggml_quantize_chunk(type, test_data, first.data(), 0, 1, test_size, imatrix.data());
    const size_t second_size = ggml_quantize_chunk(type, test_data, second.data(), 0, 1, test_size, imatrix.data());
    if (first_size != row_size || second_size != row_size || memcmp(first.data(), second.data(), row_size) != 0) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float>       dequantized(test_size);
    traits->to_float(first.data(), dequantized.data(), test_size);
    for (float value : dequantized) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool zero_weight_fallback_matches_zero_input(ggml_type     type,
                                                    size_t        subgroup_size,
                                                    size_t        test_size,
                                                    const float * test_data) {
    const size_t         row_size = ggml_row_size(type, test_size);
    std::vector<uint8_t> zero_weight_output(row_size);
    std::vector<uint8_t> zero_input_output(row_size);
    std::vector<float>   mixed_imatrix(test_size, 1.0f);
    std::vector<float>   fallback_input(test_data, test_data + test_size);
    if (type == GGML_TYPE_IQ3_S) {
        for (size_t i = 0; i < 4; ++i) {
            fallback_input[i] = 0.0f;
        }
    }
    std::vector<float> zero_input(fallback_input);

    for (size_t i = 0; i < subgroup_size; ++i) {
        mixed_imatrix[i] = 0.0f;
        zero_input[i]    = 0.0f;
    }
    std::vector<float> unit_imatrix(test_size, 1.0f);

    ggml_quantize_chunk(type, fallback_input.data(), zero_weight_output.data(), 0, 1, test_size, mixed_imatrix.data());
    ggml_quantize_chunk(type, zero_input.data(), zero_input_output.data(), 0, 1, test_size, unit_imatrix.data());

    switch (type) {
        case GGML_TYPE_IQ2_XXS:
            {
                const block_iq2_xxs * zero_weight = (const block_iq2_xxs *) zero_weight_output.data();
                const block_iq2_xxs * expected    = (const block_iq2_xxs *) zero_input_output.data();
                return memcmp(zero_weight->qs, expected->qs, subgroup_size / 8) == 0;
            }
        case GGML_TYPE_IQ2_XS:
            {
                const block_iq2_xs * zero_weight = (const block_iq2_xs *) zero_weight_output.data();
                const block_iq2_xs * expected    = (const block_iq2_xs *) zero_input_output.data();
                return ((zero_weight->qs[0] ^ expected->qs[0]) & 0x01ff) == 0 &&
                       ((zero_weight->qs[1] ^ expected->qs[1]) & 0x01ff) == 0;
            }
        case GGML_TYPE_IQ3_XXS:
            {
                const block_iq3_xxs * zero_weight = (const block_iq3_xxs *) zero_weight_output.data();
                const block_iq3_xxs * expected    = (const block_iq3_xxs *) zero_input_output.data();
                return memcmp(zero_weight->qs, expected->qs, subgroup_size / 4) == 0;
            }
        case GGML_TYPE_IQ3_S:
            {
                const block_iq3_s * zero_weight      = (const block_iq3_s *) zero_weight_output.data();
                const uint16_t      first_grid_index = zero_weight->qs[0] | ((zero_weight->qh[0] & 1) << 8);
                return first_grid_index == 0;
            }
        case GGML_TYPE_IQ2_S:
            {
                const block_iq2_s * zero_weight = (const block_iq2_s *) zero_weight_output.data();
                const block_iq2_s * expected    = (const block_iq2_s *) zero_input_output.data();
                return memcmp(zero_weight->qs, expected->qs, 2) == 0 &&
                       ((zero_weight->qh[0] ^ expected->qh[0]) & 0x0f) == 0;
            }
        case GGML_TYPE_IQ1_S:
            {
                const block_iq1_s * zero_weight = (const block_iq1_s *) zero_weight_output.data();
                const block_iq1_s * expected    = (const block_iq1_s *) zero_input_output.data();
                return memcmp(zero_weight->qs, expected->qs, subgroup_size / 8) == 0 &&
                       ((zero_weight->qh[0] ^ expected->qh[0]) & 0x8fff) == 0;
            }
        case GGML_TYPE_IQ1_M:
            {
                const block_iq1_m * zero_weight = (const block_iq1_m *) zero_weight_output.data();
                const block_iq1_m * expected    = (const block_iq1_m *) zero_input_output.data();
                return memcmp(zero_weight->qs, expected->qs, subgroup_size / 8) == 0 &&
                       zero_weight->qh[0] == expected->qh[0];
            }
        default:
            return false;
    }
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    ggml_cpu_init();

    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            float       max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR;
            switch (type) {
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_IFAIRY:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_TERNARY;
                    break;
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_IQ2_S:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_2BITS;
                    break;
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_IQ3_S:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_3BITS;
                    break;
                case GGML_TYPE_IQ3_XXS:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS;
                    break;
                default:
                    break;
            }
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            float       max_allowed_error = MAX_DOT_PRODUCT_ERROR;
            if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S) {
                max_allowed_error = MAX_DOT_PRODUCT_ERROR_LOWBIT;
            } else if (type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_IFAIRY) {
                max_allowed_error = MAX_DOT_PRODUCT_ERROR_TERNARY;
            }
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    for (ggml_type type : {
             GGML_TYPE_IQ2_XXS,
             GGML_TYPE_IQ2_XS,
             GGML_TYPE_IQ3_XXS,
             GGML_TYPE_IQ1_S,
             GGML_TYPE_IQ4_NL,
             GGML_TYPE_IQ3_S,
             GGML_TYPE_IQ2_S,
             GGML_TYPE_IQ4_XS,
             GGML_TYPE_IQ1_M,
         }) {
        const bool ok = mixed_zero_imatrix_quantization_is_deterministic(type, test_size, test_data.data());
        num_failed += !ok;
        if (!ok || verbose) {
            printf("%5s mixed-zero imatrix determinism:    %s\n", ggml_type_name(type), ok ? "ok" : "FAILED");
        }
    }

    for (const auto & test : {
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ2_XXS, 32 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ2_XS,  16 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ3_XXS, 32 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ3_S,   32 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ2_S,   16 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ1_S,   32 },
             std::pair<ggml_type, size_t>{ GGML_TYPE_IQ1_M,   16 },
    }) {
        const bool ok = zero_weight_fallback_matches_zero_input(test.first, test.second, test_size, test_data.data());
        num_failed += !ok;
        if (!ok || verbose) {
            printf("%5s zero-weight fallback encoding:    %s\n", ggml_type_name(test.first), ok ? "ok" : "FAILED");
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
