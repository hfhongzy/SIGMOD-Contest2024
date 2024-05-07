#include "utils.h"


#include <algorithm>
#include <cstring>
#include <set>

using std::pair;

void ReadData(const string &dataset_path,
                           std::map<int32_t, int>& C_count,
                           std::map<int32_t, vector<float>>& C_time_list,
                           std::map<int32_t, std::unique_ptr<float[]>>& C_data,
                           std::map<int32_t , vector<int>>& C_ids_list,
                           std::vector<std::pair<int, int>>& C_offset_vec) {
    int data_dim = 100;
    C_count.clear();
    C_time_list.clear();
    C_data.clear();
    C_ids_list.clear();
    std::vector<std::vector<float>> data;
    ReadBin(dataset_path, 102, data);
    int data_size = data.size();
    cout << "data_size = " << data_size << endl;

    vector<int> ids(data_size);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int a, int b){
        return data[a][1] < data[b][1] || (data[a][1] == data[a][1] && data[a][0] < data[a][0]);
    });

    vector<float> time(data_size);
    vector<int> _idx(data_size);
    for (int i = 0; i < data_size; i++) {
        int32_t c = data[ids[i]][0];
        _idx[ids[i]] = C_count[c]++;
    }
    for(auto& p : C_count) {
        vector<float> vt(p.second);
        vector<int> vi(p.second);
        C_time_list[p.first] = std::move(vt);
        C_ids_list[p.first] = std::move(vi);
        C_data[p.first] = std::make_unique<float[]>(p.second * data_dim);
    }

    C_data[orderByC] = std::make_unique<float[]>(data_size * 100);
    auto &order_by_C_data = C_data[orderByC];
    C_ids_list[orderByC] = vector<int>(data_size);
    auto &order_by_C_ids = C_ids_list[orderByC];
    C_time_list[orderByC] = vector<float>(data_size);
    auto &order_by_C_time = C_time_list[orderByC];

    C_data[orderByT] = std::make_unique<float[]>(data_size * data_dim);
    auto &order_by_T_data = C_data[orderByT];
#pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < data_size; i++) {
        int id = ids[i];
        int32_t c = data[id][0];
        time[i] = data[id][1];
        std::memcpy(order_by_T_data.get() + i * 100, data[id].data() + 2, 100 * 4);
        C_time_list[c][_idx[id]] = data[id][1];
        std::memcpy(C_data[c].get() + _idx[id] * 100, data[id].data() + 2, 100 * 4);
        C_ids_list[c][_idx[id]] = id;
    }

    vector<pair<int, int>> countVec(C_count.begin(), C_count.end());
    std::sort(countVec.begin(), countVec.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });
    vector<int> idx(countVec.size());
    for (int i = 1;  i < (int)countVec.size(); i++) {
        idx[i] = idx[i - 1] + countVec[i - 1].second;
    }
#pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < (int)countVec.size(); i++) {
        int c = countVec[i].first;
        std::memcpy(order_by_C_ids.data() + idx[i], C_ids_list[c].data(), C_ids_list[c].size() * sizeof(int));
        std::memcpy(order_by_C_time.data() + idx[i], C_time_list[c].data(), C_time_list[c].size() * sizeof(float));
        std::memcpy(order_by_C_data.get() + idx[i] * 100, C_data[c].get(), countVec[i].second * 100 * 4);
    }
    assert(order_by_C_ids.size() == data_size);
    assert(order_by_C_time.size() == data_size);
    // assert(idx == data_size);
    C_ids_list[orderByT] = std::move(ids);
    C_time_list[orderByT] = std::move(time);
    C_count[orderByT] = data_size;
    C_count[orderByC] = data_size;

    C_offset_vec.clear();
    int offset = 0;
    for(auto [c, num] : countVec) {
        C_offset_vec.emplace_back(std::make_pair(c, offset));
        offset += num;
    }
    for(int i = 0; i < 10; i++) {
        cout << "c = " << C_offset_vec[i].first << ", num = " << C_count[C_offset_vec[i].first] << ", offset = " << C_offset_vec[i].second << endl;
    }
}

void compute_recall(const string& gt_file, vector<vector<float>>& queries, vector<vector<int>>& output,
                    std::map<int32_t, vector<int>>& C_brute_queries_ids, std::map<int32_t, vector<int>>& C_graph_queries_ids) {
    vector<vector<int>> GT{};
    ReadBin(gt_file, 100, GT);
    int total_coselection = 0;
    vector<int> type_num(4, 0);
    vector<int> each_type_coselection(4, 0);
    int subsection_num = 4;
    float subsection_range = 1.0f / subsection_num;
    vector<int> t2_subsection_num(subsection_num);
    vector<int> t2_subsection_coselection(subsection_num);
    int nq = queries.size();
    int tmp = 0;
    for (int i = 0; i < nq; i++) {
//        assert(output[i].size() == 100);
//        if(output[i].size() != 100) cout  << "i = " << i << ", output[i].size() != 100" << endl;
//        if(output[i].size() == 0 ) continue;
        uint32_t query_type = queries[i][0];
        int32_t c = queries[i][1];
        float l = queries[i][2], r = queries[i][3];
        int t2_subsection_idx = 0;
        if(query_type == 2) {
            for(int j = 0; j < subsection_num; j++){
                if(r - l > j * subsection_range) {
                    t2_subsection_idx = j;
                }
            }
        }

        type_num[query_type]++;
        int cur_coselection = 0;
        std::set gt(GT[i].begin(), GT[i].end());
        std::set res(output[i].begin(), output[i].end());
        for (auto item: res) {
            if (gt.find(item) != gt.end()) {
                cur_coselection++;
            }
        }
        total_coselection += cur_coselection;
        each_type_coselection[query_type] += cur_coselection;

        if(query_type == 2) {
            t2_subsection_coselection[t2_subsection_idx] += cur_coselection;
            t2_subsection_num[t2_subsection_idx]++;
        }
    }

    int t1_graph_num = 0, t1_graph_coselection = 0;
    int t1_brute_num = 0, t1_brute_coselection = 0;
    std::map<int32_t, int> C_t1_graph_num_map{};
    std::map<int32_t, int> C_t1_coselection_num_map{};

    std::map<int32_t, int> C_t3_graph_num_map{};
    std::map<int32_t, int> C_t3_coselection_num_map{};

    int t3_graph_num = 0, t3_graph_coselection = 0;
    int t3_brute_num = 0, t3_brute_coselection = 0;

    for(auto [c, queries_ids] : C_graph_queries_ids) {
        for(auto qid : queries_ids) {
            uint32_t query_type = queries[qid][0];
            int32_t c = queries[qid][1];
            float l = queries[qid][2], r = queries[qid][3];
            if(query_type == 1) {
                C_t1_graph_num_map[c]++;
                t1_graph_num++;
            } else if(query_type == 3) {
                C_t3_graph_num_map[c]++;
                t3_graph_num++;
            }
            int cur_coselection = 0;
            std::set gt(GT[qid].begin(), GT[qid].end());
            std::set res(output[qid].begin(), output[qid].end());
            for (auto item: res) {
                if (gt.find(item) != gt.end()) {
                    cur_coselection++;
                }
            }
            if(query_type == 1) {
                C_t1_coselection_num_map[c] += cur_coselection;
                t1_graph_coselection += cur_coselection;
            } else if(query_type == 3) {
                C_t3_coselection_num_map[c] += cur_coselection;
                t3_graph_coselection += cur_coselection;
            }
        }
    }

    for(auto [c, queries_ids] : C_brute_queries_ids) {
        for(auto qid : queries_ids) {
            uint32_t query_type = queries[qid][0];
            int32_t c = queries[qid][1];
            float l = queries[qid][2], r = queries[qid][3];
            if(query_type == 1) {
                t1_brute_num++;
            } else if(query_type == 3) {
                t3_brute_num++;
            }
            int cur_coselection = 0;
            std::set gt(GT[qid].begin(), GT[qid].end());
            std::set res(output[qid].begin(), output[qid].end());
            for (auto item: res) {
                if (gt.find(item) != gt.end()) {
                    cur_coselection++;
                }
            }
            if(query_type == 1) {
                t1_brute_coselection += cur_coselection;
            } else if(query_type == 3) {
                t3_brute_coselection += cur_coselection;
            }
        }
    }

    std::cout << "recall = " << (double) total_coselection * 100 / (nq * 100) << " %" << std::endl;
    for (int i = 0; i < 4; i++) {
        cout << "query type = " << i << ", reacll = "
             << (double) each_type_coselection[i] * 100 / (type_num[i] * 100) << " %" << endl;
    }
    cout << "query type = " << 1 << ", graph search reacll = "
         << (double) t1_graph_coselection * 100 / (t1_graph_num * 100) << " %" << endl;
    cout << "query type = " << 1 << ", brute force search reacll = "
         << (double) t1_brute_coselection * 100 / (t1_brute_num * 100) << " %" << endl;
    for(auto [c, num] : C_t1_graph_num_map) {
        cout << "query type = 1, C = " << c << ", queries_num = " << num << ", graph search reacll = "
             << (double) C_t1_coselection_num_map[c] * 100 / (C_t1_graph_num_map[c] * 100) << " %" << endl;
    }
    for(int i = 0; i < subsection_num; i++) {
        cout << "query type = 2, range [" << i * subsection_range << ", " << (i + 1) * subsection_range << "], num = "<< t2_subsection_num[i] <<", recall = " <<  (double) t2_subsection_coselection[i] * 100 / (t2_subsection_num[i] * 100) << " %" << endl;
    }
    cout << "query type = " << 3 << ", graph search reacll = "
         << (double) t3_graph_coselection * 100 / (t3_graph_num * 100) << " %" << endl;
    cout << "query type = " << 3 << ", brute force search reacll = "
         << (double) t3_brute_coselection * 100 / (t3_brute_num * 100) << " %" << endl;
    for(auto [c, num] : C_t3_graph_num_map) {
        cout << "query type = 3, C = " << c << ", queries_num = " << num << ", graph search reacll = "
             << (double) C_t3_coselection_num_map[c] * 100 / (C_t3_graph_num_map[c] * 100) << " %" << endl;
    }

}

void instruction_set() {
#if defined(__AVX512F__)
    cout << "__AVX512F__" << endl;
#endif
#if defined(__AVX2__)
    cout << "__AVX2__" << endl;
#endif
#if defined(__aarch64__)
    cout << "__aarch64__" << endl;
#endif
#if defined(__SSE2__)
    cout << "__SSE2__" << endl;
#endif
}