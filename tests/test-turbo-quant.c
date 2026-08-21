#include "ggml.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void quantize_row_turbo2_0_ref(const float * x, void * y, long long k);
extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void turbo_cpu_fwht_forward(float * x, int group_size);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

static const uint8_t source_turbo2[34] = {
    0xb7, 0x4e, 0x8e, 0x59, 0xe4, 0xca, 0x56, 0x59, 0x64, 0xa6, 0xe2, 0xeb, 0x69, 0xae, 0xae, 0xb5, 0x9e,
    0x62, 0x16, 0xa2, 0x87, 0xe8, 0xb6, 0x3d, 0x5c, 0x61, 0x57, 0x55, 0xc9, 0x75, 0x24, 0xd6, 0xd8, 0xb9,
};

static const uint8_t source_turbo3[50] = {
    0x27, 0x4e, 0x19, 0xb6, 0x88, 0xd0, 0xe9, 0xe7, 0xd9, 0x4c, 0x88, 0xd7, 0x87, 0x09, 0x58, 0x2b, 0x78,
    0x90, 0x7c, 0x51, 0x5e, 0xd0, 0x2d, 0x5a, 0xed, 0xd7, 0xae, 0xfa, 0x86, 0xae, 0x4a, 0xe8, 0xa1, 0x66,
    0x2b, 0xbc, 0x21, 0xd4, 0xfd, 0xf6, 0xcf, 0x5b, 0xd1, 0xe9, 0x6d, 0x42, 0x01, 0x4a, 0x94, 0xea,
};

static const uint8_t source_turbo4[66] = {
    0x34, 0x4e, 0xbb, 0x83, 0xa6, 0x56, 0x42, 0xc8, 0x89, 0xd2, 0x4a, 0x65, 0xa7, 0x64, 0x54, 0x7b, 0x69,
    0xb9, 0x49, 0xc9, 0xbe, 0xdb, 0xa7, 0x48, 0xdb, 0x99, 0xb9, 0xba, 0x56, 0x8d, 0xd8, 0xa6, 0x28, 0x4a,
    0x79, 0x26, 0x2a, 0xaa, 0x6d, 0xa3, 0x80, 0xda, 0x7a, 0x9d, 0xb4, 0x2b, 0xe3, 0x76, 0x26, 0x7a, 0x7c,
    0x45, 0x54, 0x66, 0xa5, 0xb1, 0x75, 0x4c, 0x64, 0x29, 0x69, 0xf4, 0x82, 0xc4, 0xb5, 0xac,
};

static void fill_input(float * input, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        input[i] = sinf((float) i * 0.137f) * 3.0f + cosf((float) i * 0.071f) * 0.25f;
    }
}

static int check_source_bytes(void) {
    float   input[128];
    uint8_t turbo2[sizeof(source_turbo2)];
    uint8_t turbo3[sizeof(source_turbo3)];
    uint8_t turbo4[sizeof(source_turbo4)];

    fill_input(input, 128);
    quantize_row_turbo2_0_ref(input, turbo2, 128);
    quantize_row_turbo3_0_ref(input, turbo3, 128);
    quantize_row_turbo4_0_ref(input, turbo4, 128);

    return memcmp(turbo2, source_turbo2, sizeof(turbo2)) != 0 || memcmp(turbo3, source_turbo3, sizeof(turbo3)) != 0 ||
           memcmp(turbo4, source_turbo4, sizeof(turbo4)) != 0;
}

static int check_wht_roundtrip(void) {
    float  input[512];
    float  rotated[512];
    double err = 0.0;
    double ref = 0.0;

    fill_input(input, 512);
    memcpy(rotated, input, sizeof(input));
    for (int i = 0; i < 512; i += 128) {
        turbo_cpu_fwht_forward(rotated + i, 128);
        turbo_cpu_fwht_inverse(rotated + i, 128);
    }
    for (int i = 0; i < 512; ++i) {
        const double diff = (double) rotated[i] - (double) input[i];
        err += diff * diff;
        ref += (double) input[i] * (double) input[i];
    }

    const double nmse = err / ref;
    printf("WHT roundtrip NMSE: %.9g\n", nmse);
    return nmse > 1e-5;
}

static int check_chunked_dequant(enum ggml_type type, int64_t n) {
    const struct ggml_type_traits * traits    = ggml_get_type_traits(type);
    const int64_t                   block     = traits->blck_size;
    const int64_t                   chunk     = (256 / block) * block;
    float *                         input     = malloc((size_t) n * sizeof(float));
    void *                          quantized = malloc(ggml_row_size(type, n));
    float *                         whole     = malloc((size_t) n * sizeof(float));
    float *                         pieces    = malloc((size_t) n * sizeof(float));

    fill_input(input, n);
    ggml_quantize_chunk(type, input, quantized, 0, 1, n, NULL);
    traits->to_float(quantized, whole, n);

    const char * src = quantized;
    for (int64_t i = 0; i < n; i += chunk) {
        const int64_t count = chunk < n - i ? chunk : n - i;
        traits->to_float(src, pieces + i, count);
        src += (count / block) * traits->type_size;
    }

    const int failed = memcmp(whole, pieces, (size_t) n * sizeof(float)) != 0;
    free(input);
    free(quantized);
    free(whole);
    free(pieces);
    return failed;
}

int main(void) {
    int failures = 0;
    failures += check_source_bytes();
    failures += check_wht_roundtrip();

    const enum ggml_type types[] = {
        GGML_TYPE_TURBO2_0,
        GGML_TYPE_TURBO3_0,
        GGML_TYPE_TURBO4_0,
    };

    const int64_t sizes[] = { 128, 256, 384, 512, 4096 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); ++t) {
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
            failures += check_chunked_dequant(types[t], sizes[i]);
        }
    }

    if (failures != 0) {
        fprintf(stderr, "TurboQuant test failures: %d\n", failures);
        return 1;
    }
    printf("TurboQuant source bytes and chunked dequant: OK\n");
    return 0;
}
