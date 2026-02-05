#pragma once

#include "common.cuh"

void ggml_cuda_op_ifairy_rmsnorm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_split(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ifairy_relu2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
