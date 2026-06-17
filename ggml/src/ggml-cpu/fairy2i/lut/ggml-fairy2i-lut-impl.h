#pragma once

#include "ggml-fairy2i-lut.h"

static const int    k_fairy2i_lut_entries     = 16;
static const int    k_fairy2i_lut_channels    = 4;
static const size_t k_fairy2i_lut_group_bytes = (size_t) k_fairy2i_lut_channels * (size_t) k_fairy2i_lut_entries;
