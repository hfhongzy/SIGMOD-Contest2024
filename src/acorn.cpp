#include <iostream>
#include <chrono>
#include <filesystem>
#include <set>

#include "io.h"
#include "common.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "neighbor.h"
#include "quant/fp32_quant.h"
#include "quant/sq8_quant.h"
#include <omp.h>
#include <hash_table7.hpp>
using std::cout;
using std::endl;
namespace fs = std::filesystem;
using std::pair;

int main() {
    cout << "acorn" << endl;
    instruction_set();
    omp_set_num_threads(32);
    auto start = std::chrono::high_resolution_clock::now();
    std::map<int32_t, int> C_count{};
    std::map<int32_t, vector<float>> C_time_list{};
    std::map<int32_t, std::unique_ptr<float[]>> C_data{};
    std::map<int32_t , vector<int>> C_ids_list{};
    std::vector<std::pair<int, int>> C_offset_vec{};
    std::map<int32_t, hnswlib::HierarchicalNSW<float>* > C_HNSW{};
    std::map<int32_t, FP32Quantizer*> C_fp32{};
    std::map<int32_t, SQ8Quantizer*> C_sq8{};
    hnswlib::L2Space space(100);

    ReadData(dummy_data, C_count, C_time_list, C_data, C_ids_list, C_offset_vec);
    for(auto& [c, data] : C_data) {
        int n = C_count[c];
        C_fp32[c] = new FP32Quantizer(100);
        C_fp32[c]->train(data.get(), n);
        C_sq8[c] = new SQ8Quantizer(100);
        C_sq8[c]->train(data.get(), n);
    }

    auto end = std::chrono::high_resolution_clock::now();
    cout  <<"load data time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    int nb = C_count[orderByC], d = 100;
    cout << "nb = " << nb << endl;

    bool get_recall = false;
    bool save_output = true;
    bool update_index = true;
    bool save_index = false;


    int M = 20;
    int R = 2 * M;
    int L = 200;
    int ef = 1200;
    int dim = 100;
    int brute_force_threshold[] = {0, 450'000, 450'000, 450'000};
    int sub_graph_num = 3;
    assert(sub_graph_num >= 1);
    int candidate_num = 150;
    assert(candidate_num >= 100 && candidate_num <= ef);

    start = std::chrono::high_resolution_clock::now();
    if(update_index) {
        int max_C = C_offset_vec.front().first;
        int num = C_count[max_C];
        float* data = C_data[orderByC].get();
        C_HNSW.emplace(orderByC, new hnswlib::HierarchicalNSW<float>(&space, nb, M, L, C_sq8[orderByC]));
        auto orderByC_index = C_HNSW[orderByC];
        auto build_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num; i++) {
            orderByC_index->addPoint(data + i * 100, i);
        }
        C_HNSW.emplace(max_C, new hnswlib::HierarchicalNSW<float>(&space, nb, M, L, C_sq8[orderByC], *orderByC_index));
        std::vector<std::pair<hnswlib::HierarchicalNSW<float>*, int>> pairs(2 * nb);
#pragma omp parallel for schedule(dynamic)
        for (int i = num; i < nb; i++) {
            pairs[i - num] = std::make_pair(orderByC_index, i);
        }
        int idx = nb - num;
        for(int i = 1; i < sub_graph_num; i++) {
            int c = C_offset_vec[i].first;
            int sub_num = C_count[c];
            C_HNSW.emplace(c, new hnswlib::HierarchicalNSW<float>(&space, sub_num, M, L, C_sq8[c]));
            auto sub_index = C_HNSW.at(c);
#pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < sub_num; j++) {
                pairs[idx + j] = std::make_pair(sub_index, j);
            }
            idx += sub_num;
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < idx; i++) {
            pairs[i].first->addPoint(nullptr, pairs[i].second);
        }
        // cout << "[full graph] R = " << R << ", L = " << L <<", build time: " << std::chrono::duration<double>(build_end - build_start).count() << " s" << std::endl;
        if (save_index) {
            orderByC_index->saveIndex(index_file + std::to_string(orderByC));
            C_HNSW.at(max_C)->saveIndex(index_file + std::to_string(max_C));
            for(int i = 1; i < sub_graph_num; i++) {
                int c = C_offset_vec[i].first;
                auto sub_index = C_HNSW.at(c);
                sub_index->saveIndex(index_file + std::to_string(c));
            }
        }
        auto build_end = std::chrono::high_resolution_clock::now();
        cout << "build time: " << std::chrono::duration<double>(build_end - build_start).count() << " s" << std::endl;
    } else {
        if(!fs::exists(index_file + std::to_string(orderByC))) cout << "hnsw index isn't existed." << endl;
        C_HNSW[orderByC] = new hnswlib::HierarchicalNSW<float>(&space, index_file + std::to_string(orderByC), C_sq8[orderByC]);
        for(int i = 0; i < sub_graph_num; i++) {
            int c = C_offset_vec[i].first;
            int offset = C_offset_vec[i].second;
            int num = C_count[c];
            if(!fs::exists(index_file + std::to_string(c))) cout << "hnsw index isn't existed." << endl;
            if (i != 0) {
                C_HNSW[c] = new hnswlib::HierarchicalNSW<float>(&space, index_file + std::to_string(c), C_sq8[c]);
            } else {
                C_HNSW[c] = new hnswlib::HierarchicalNSW<float>(&space, index_file + std::to_string(c), C_sq8[orderByC]);
            }
        }
    }
    for (auto &[c, index] : C_HNSW) {
        index->time_list = C_time_list[c];
        index->SortEdgesLayer0();
    }
    cout << endl;
    end = std::chrono::high_resolution_clock::now();
    cout << "R = " << R << ", L = " << L <<", build time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector<vector<float>> queries{};
    ReadBin(dummy_queries, 104, queries);
    int queries_num = queries.size();
    cout << "queries_num = " << queries_num << endl;
    vector<int> q_brute, q_graph, type_id;
    vector<std::pair<int, int>> g_interval, b_interval;
    {
        vector<int> buc[4];
        for (int i = 0; i < queries_num; i++) {
            int32_t c = queries[i][0];
            buc[c].emplace_back(i);
        }
        for (int i = 0; i < 4; i++)
            for (int j : buc[i]) type_id.emplace_back(j);
    }
    for (int t = 0; t < queries_num; t++) {
        int i = type_id[t];
        uint32_t query_type = queries[i][0];
        int32_t c = queries[i][1];
        float l = queries[i][2], r = queries[i][3];
        if(query_type == 0) {
            q_graph.emplace_back(i);
            g_interval.emplace_back(0, 0);
        } else if(query_type == 1) {
            if(C_HNSW.find(c) != C_HNSW.end() && C_count[c] > brute_force_threshold[1]) {
                q_graph.emplace_back(i);
                g_interval.emplace_back(0, 0);
            }
            else {
                q_brute.emplace_back(i);
                b_interval.emplace_back(0, 0);
            }
        } else if(query_type == 2) {
            auto& time_list = C_time_list[orderByT];
            int lower = std::lower_bound(time_list.begin(), time_list.end(), l) - time_list.begin();
            int upper = std::upper_bound(time_list.begin(), time_list.end(), r) - time_list.begin() - 1;
            if((upper - lower + 1) > brute_force_threshold[2]) {
                q_graph.emplace_back(i);
                g_interval.emplace_back(lower, upper);
            } else {
                q_brute.emplace_back(i);
                b_interval.emplace_back(lower, upper);
            }
        } else {
            auto& time_list = C_time_list[c];
            int lower = std::lower_bound(time_list.begin(), time_list.end(), l) - time_list.begin();
            int upper = std::upper_bound(time_list.begin(), time_list.end(), r) - time_list.begin() - 1;
            if(C_HNSW.find(c) != C_HNSW.end()) {
                if((upper - lower + 1) > brute_force_threshold[3]) {
                    q_graph.emplace_back(i);
                    g_interval.emplace_back(lower, upper);
                } else {
                    q_brute.emplace_back(i);
                    b_interval.emplace_back(lower, upper);
                }
            }
            else {
                q_brute.emplace_back(i);
                b_interval.emplace_back(lower, upper);
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    cout << "load query time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    vector<vector<int>> output(queries_num);
    auto &T_total_sq8 = C_sq8[orderByT];
    auto &T_total_fp32 = C_fp32[orderByT];
    auto &T_total_ids_list = C_ids_list[orderByT];
    vector<LPool<float, int>> _re_rank_pool(128, LPool<float, int>(100));
{
    int8_t *sq[128];
    float *fp[128];
    for (int i = 0; i < 128; i++) {
        sq[i] = (int8_t *)alloc64B(112);
        fp[i] = (float *)alloc64B(112 * 4);
    }
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)q_graph.size(); t++) {
        int qid = q_graph[t], tid = omp_get_thread_num();
        auto &re_rank_pool = _re_rank_pool[tid];
        re_rank_pool.size_ = re_rank_pool.cur_ = 0;
        uint32_t query_type = queries[qid][0];
        uint32_t c = (query_type == 0 || query_type == 2) ? orderByC : queries[qid][1];
        auto& ids = C_ids_list[c];
        auto c_fp32_quant = C_fp32[c];
        float* q = queries[qid].data() + 4;
        c_fp32_quant->encode(q, (char *)fp[tid]);
        auto index = C_HNSW[c];
        float l = queries[qid][2], r = queries[qid][3];
        int8_t *temp_q = sq[tid];
        index->quant_->encode(q, (char*)temp_q);
        int candidate_num = 150;
        auto pool = index->searchKnn_filter(temp_q, candidate_num, query_type, 0, l, r, 
                                            g_interval[t].second - g_interval[t].first + 1);
        #ifdef USE_SSE
            _mm_prefetch(c_fp32_quant->get_data(pool.pool[0].second), _MM_HINT_T0);
            _mm_prefetch(c_fp32_quant->get_data(pool.pool[0].second) + 64, _MM_HINT_T0);
        #endif
        for (int j = 0; j < pool.size(); j++) {
            int id = pool.pool[j].second;
            if (j + 1 < pool.size()) {
                #ifdef USE_SSE
                    _mm_prefetch(c_fp32_quant->get_data(pool.pool[j + 1].second), _MM_HINT_T0);
                    _mm_prefetch(c_fp32_quant->get_data(pool.pool[j + 1].second) + 64, _MM_HINT_T0);
                #endif
            }
            re_rank_pool.emplace(L2SqrRef((float *)c_fp32_quant->get_data(id), fp[tid], 112), id);
        }
        vector<int> res(100);
        for (int j = 0; j < std::min(100, (int)re_rank_pool.size_); j++) {
            res[j] = ids[re_rank_pool.id(j)];
        }
        if (re_rank_pool.size_ != 100) {
            cout << "graph search result.size() != 100" << endl;
        }
        output[qid] = std::move(res);
    }
}
    end = std::chrono::high_resolution_clock::now();
    cout << "graph search time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector<LPool<int, int>> _pool(128, LPool<int, int>(candidate_num));
{
    int8_t *sq[128];
    float *fp[128];
    for (int i = 0; i < 128; i++) {
        sq[i] = (int8_t *)alloc64B(112);
        fp[i] = (float *)alloc64B(112 * 4);
    }
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)q_brute.size(); t++) {
        int qid = q_brute[t], tid = omp_get_thread_num();
        uint32_t query_type = queries[qid][0];
        uint32_t c = (query_type == 0 || query_type == 2) ? orderByT : queries[qid][1];
        output[qid].resize(100);
        auto &re_rank_pool = _re_rank_pool[tid];
        re_rank_pool.size_ = re_rank_pool.cur_ = 0;
        auto &pool = _pool[tid];
        pool.size_ = 0;
        if (query_type == 1) {
            auto& ids_list = C_ids_list[c];
            auto c_fp32_quant = C_fp32[c];
            auto c_sq8_quant = C_sq8[c];
            int c_nb = C_count[c];
            float* q = queries[qid].data() + 4;
            c_sq8_quant->encode(q, (char *)sq[tid]);
            c_fp32_quant->encode(q, (char *)fp[tid]);
            int lower = 0, upper = c_nb - 1;
            #ifdef USE_SSE
            _mm_prefetch(c_sq8_quant->get_data(lower), _MM_HINT_T0);
            _mm_prefetch(c_sq8_quant->get_data(lower) + 64, _MM_HINT_T0);
            #endif
            for(int j = lower; j <= upper; j++) {
                if (j + 1 <= upper) {
                    #ifdef USE_SSE
                    _mm_prefetch(c_sq8_quant->get_data(j + 1), _MM_HINT_T0);
                    _mm_prefetch(c_sq8_quant->get_data(j + 1) + 64, _MM_HINT_T0);
                    #endif
                }
                pool.emplace(L2SqrSQ8((int8_t *)c_sq8_quant->get_data(j), sq[tid], 112), j);
            }
            #ifdef USE_SSE
                _mm_prefetch(c_fp32_quant->get_data(pool.id(0)), _MM_HINT_T0);
                _mm_prefetch(c_fp32_quant->get_data(pool.id(0)) + 64, _MM_HINT_T0);
            #endif
            for(int j = 0; j < pool.size_; j++) {
                int id = pool.id(j);
                if (j + 1 < pool.size_) {
                #ifdef USE_SSE
                    _mm_prefetch(c_fp32_quant->get_data(pool.id(j + 1)), _MM_HINT_T0);
                    _mm_prefetch(c_fp32_quant->get_data(pool.id(j + 1)) + 64, _MM_HINT_T0);
                #endif
                }
                re_rank_pool.emplace(L2SqrRef((float *)c_fp32_quant->get_data(id), fp[tid], 112), id);
            }
            for(int j = 0; j < 100; j++) {
                output[qid][j] = ids_list[re_rank_pool.id(j)];
            }
        } else if (query_type == 2) {
            float* q = queries[qid].data() + 4;
            T_total_sq8->encode(q, (char *)sq[tid]);
            T_total_fp32->encode(q, (char *)fp[tid]);
            int lower = b_interval[t].first;
            int upper = b_interval[t].second;
            #ifdef USE_SSE
            _mm_prefetch(T_total_sq8->get_data(lower), _MM_HINT_T0);
            _mm_prefetch(T_total_sq8->get_data(lower) + 64, _MM_HINT_T0);
            #endif
            for (int j = lower; j <= upper; j++) {
                if (j + 1 <= upper) {
                    #ifdef USE_SSE
                    _mm_prefetch(T_total_sq8->get_data(j + 1), _MM_HINT_T0);
                    _mm_prefetch(T_total_sq8->get_data(j + 1) + 64, _MM_HINT_T0);
                    #endif
                }
                pool.emplace(L2SqrSQ8((int8_t *)T_total_sq8->get_data(j), sq[tid], 112), j);
            }
            #ifdef USE_SSE
                _mm_prefetch(T_total_fp32->get_data(pool.id(0)), _MM_HINT_T0);
                _mm_prefetch(T_total_fp32->get_data(pool.id(0)) + 64, _MM_HINT_T0);
            #endif
            for(int j = 0; j < pool.size_; j++) {
                int id = pool.id(j);
                if (j + 1 < pool.size_) {
                #ifdef USE_SSE
                    _mm_prefetch(T_total_fp32->get_data(pool.id(j + 1)), _MM_HINT_T0);
                    _mm_prefetch(T_total_fp32->get_data(pool.id(j + 1)) + 64, _MM_HINT_T0);
                #endif
                }
                re_rank_pool.emplace(L2SqrRef((float *)T_total_fp32->get_data(id), fp[tid], 112), id);
            }
            for(int j = 0; j < 100; j++) {
                output[qid][j] = T_total_ids_list[re_rank_pool.id(j)];
            }
        } else {
            auto c_fp32_quant = C_fp32[c];
            auto c_sq8_quant = C_sq8[c];
            auto& ids_list = C_ids_list[c];
            float* q = queries[qid].data() + 4;
            c_sq8_quant->encode(q, (char *)sq[tid]);
            c_fp32_quant->encode(q, (char *)fp[tid]);
            int lower = b_interval[t].first;
            int upper = b_interval[t].second;
            #ifdef USE_SSE
            _mm_prefetch(c_sq8_quant->get_data(lower), _MM_HINT_T0);
            _mm_prefetch(c_sq8_quant->get_data(lower) + 64, _MM_HINT_T0);
            #endif
            for(int j = lower; j <= upper; j++) {
                if (j + 1 <= upper) {
                    #ifdef USE_SSE
                    _mm_prefetch(c_sq8_quant->get_data(j + 1), _MM_HINT_T0);
                    _mm_prefetch(c_sq8_quant->get_data(j + 1) + 64, _MM_HINT_T0);
                    #endif
                }
                pool.emplace(L2SqrSQ8((int8_t *)c_sq8_quant->get_data(j), sq[tid], 112), j);
            }
            #ifdef USE_SSE
                _mm_prefetch(c_fp32_quant->get_data(pool.id(0)), _MM_HINT_T0);
                _mm_prefetch(c_fp32_quant->get_data(pool.id(0)) + 64, _MM_HINT_T0);
            #endif
            for(int j = 0; j < pool.size_; j++) {
                int id = pool.id(j);
                if (j + 1 < pool.size_) {
                #ifdef USE_SSE
                    _mm_prefetch(c_fp32_quant->get_data(pool.id(j + 1)), _MM_HINT_T0);
                    _mm_prefetch(c_fp32_quant->get_data(pool.id(j + 1)) + 64, _MM_HINT_T0);
                #endif
                }
                re_rank_pool.emplace(L2SqrRef((float *)c_fp32_quant->get_data(id), fp[tid], 112), id);
            }

            for(int j = 0; j < 100; j++) {
                output[qid][j] = ids_list[re_rank_pool.id(j)];
            }
        }
    }
}
    end = std::chrono::high_resolution_clock::now();
    cout << "brute-force search time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    if(get_recall) {
        std::map<int32_t, vector<int>> C_brute_queries_ids{};
        std::map<int32_t, vector<int>> C_graph_queries_ids{};
        compute_recall(large_gt, queries, output, C_brute_queries_ids, C_graph_queries_ids);
    }
    if(save_output) {
        SaveKNN(output);
    }
}