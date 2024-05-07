#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <immintrin.h>
#include <cstring>
#include <sys/mman.h>

#include "io.h"
#include "common.h"

using std::string;
using std::vector;
using std::unique_ptr;


void ReadData(const string &dataset_path,
              std::map<int32_t, int>& C_count,
              std::map<int32_t, vector<float>>& C_time_list,
              std::map<int32_t, std::unique_ptr<float[]>>& C_data,
              std::map<int32_t , vector<int>>& C_ids_list,
              std::vector<std::pair<int, int>>& C_offset_vec);

void compute_recall(const string& gt_file, vector<vector<float>>& queries, vector<vector<int>>& output,
                    std::map<int32_t, vector<int>>& C_brute_queries_ids, std::map<int32_t, vector<int>>& C_graph_queries_ids);

void instruction_set();

inline constexpr int64_t do_align(int64_t x, int64_t align) {
    return (x + align - 1) / align * align;
}

static inline int
L2SqrSQ8(const int8_t *x, const int8_t *y, int d) {
    __m256i res = _mm256_setzero_si256(), sub, xx_16, yy_16;
    __m128i  xx_8, yy_8;
    xx_8 = _mm_loadu_si128((__m128i *)(x + 0));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 0));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 16));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 16));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 32));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 32));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 48));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 48));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 64));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 64));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 80));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 80));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    xx_8 = _mm_loadu_si128((__m128i *)(x + 96));
    yy_8 = _mm_loadu_si128((__m128i *)(y + 96));
    xx_16 = _mm256_cvtepi8_epi16(xx_8);
    yy_16 = _mm256_cvtepi8_epi16(yy_8);
    sub = _mm256_sub_epi16(xx_16, yy_16);
    res = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), res);
    __m128i sum = _mm_add_epi32(_mm256_extracti32x4_epi32(res, 0), _mm256_extracti32x4_epi32(res, 1));
    __m128i tmp = _mm_hadd_epi32(sum, sum);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}
_Pragma("GCC push_options") _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
inline float L2SqrRef(const float *x, const float *y, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}
_Pragma("GCC pop_options")

template <typename T> struct align_alloc {
    T *ptr = nullptr;
    using value_type = T;
    T *allocate(int n) {
        if (n <= 1 << 14) {
            int sz = (n * sizeof(T) + 63) >> 6 << 6;
            return ptr = (T *)std::aligned_alloc(64, sz);
        }
        int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
        ptr = (T *)std::aligned_alloc(1 << 21, sz);
        madvise(ptr, sz, MADV_HUGEPAGE);
        return ptr;
    }
    void deallocate(T *, int) { free(ptr); }
    template <typename U> struct rebind {
        typedef align_alloc<U> other;
    };
    bool operator!=(const align_alloc &rhs) { return ptr != rhs.ptr; }
    bool operator==(const align_alloc &rhs) { return ptr == rhs.ptr; }
};

inline void *alloc2M(size_t nbytes) {
    size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
    auto p = std::aligned_alloc(1 << 21, len);
    madvise(p, len, MADV_HUGEPAGE);
    std::memset(p, 0, len);
    return p;
}

inline void *alloc64B(size_t nbytes) {
    size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
    auto p = std::aligned_alloc(1 << 6, len);
    std::memset(p, 0, len);
    return p;
}