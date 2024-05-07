/**
 * This part is adapted from https://github.com/zilliztech/pyglass/blob/master/glass/quant/sq8_quant.hpp.
 * However, instead of using minimal value and maximal value in each dimension, we use global maximal absolute
 * value in all dimensions to implement SQ8 encoding.
*/
#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include "utils.h"

struct SQ8Quantizer {
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  float global_max = 0.0f;

  SQ8Quantizer() = default;
  explicit SQ8Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        global_max(0.0f) {}

  ~SQ8Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int i = 0; i < n; i++) {
      for (int j = 0, st = i * d; j < d; j++) {
        global_max = std::max(global_max, std::fabs(data[st + j]));
      }
    }
    codes = (char *)alloc2M(n * code_size);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
      encode(data + i * d, get_data(i));
    }
  }
  void encode(const float *from, char *to) const {
    for (int i = 0; i < d; i++) {
      float x = from[i] / global_max;
      x = std::min(x, 1.0f);
      x = std::max(x, -1.0f);
      to[i] = std::round(x * 127.0f);
    }
  }
  char *get_data(int u) const { return codes + u * code_size; }
};
