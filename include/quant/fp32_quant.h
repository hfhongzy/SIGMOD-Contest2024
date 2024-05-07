/**
 * This part is adapted from https://github.com/zilliztech/pyglass/blob/master/glass/quant/fp32_quant.hpp.
 */
#pragma once
#include "utils.h"

struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  vector<float> time_list{};

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

  ~FP32Quantizer() { free(codes); }

  void train(const float *data, int64_t n) {
    codes = (char *)alloc2M(n * code_size);
// #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

  char *get_data(int64_t u) const { return codes + u * code_size; }
};
