#pragma once

#include "hnswlib.h"
#include "visited_list_pool.h"
#include "quant/sq8_quant.h"
#include "../heap.h"
#include "../hash_table7.hpp"
#include <assert.h>
#include <atomic>
#include <list>
#include <random>
#include <stdlib.h>
#include <unordered_set>
#include <fstream>
#include <vector>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
public:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count{
      0}; // current number of elements
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  mutable std::atomic<size_t> num_deleted_{0}; // number of deleted elements
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  VisitedListPool *visited_list_pool_{nullptr};

  std::mutex global;
  std::vector<std::mutex> link_list_locks_;

  // Locks operations with element by label value
  mutable std::vector<std::mutex> label_op_locks_;

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_; // keeps level of each element

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;
  DISTFUNC<int> sq8distfunc_;
  SQ8Quantizer *quant_ = nullptr;
  void *dist_func_param_{nullptr};

  mutable std::mutex label_lookup_lock; // lock for label_lookup_
  emhash7::HashMap<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  bool allow_replace_deleted_ = false; // flag to replace deleted elements
                                       // (marked as deleted) during insertions

  std::vector<float> time_list{};
  std::vector<std::pair<float, tableint>> edges;
  void SortEdgesLayer0() {
    edges.reserve((maxM0_ + 1) * max_elements_);
    edges.resize((maxM0_ + 1) * max_elements_);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)edges.size(); i += maxM0_ + 1) {
      int *data = (int *)get_linklist0(i / (maxM0_ + 1));
      int size = getListCount((linklistsizeint *)data);
      edges[i] = std::make_pair(0, size);
      for (int j = 1; j <= size; j++) {
        int id = *(data + j);
        int ei = getExternalLabel(id);
        edges[i + j] = std::make_pair(time_list[ei], id);
      }
      // std::sort(edges.begin() + i + 1, edges.begin() + i + size + 1);
    }
  }
  std::mutex deleted_elements_lock; // lock for deleted_elements
  std::unordered_set<tableint>
      deleted_elements; // contains internal ids of deleted elements

  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M,
                  size_t ef_construction, SQ8Quantizer *quant, const HierarchicalNSW<dist_t> &r,
                  size_t random_seed = 100)
      : link_list_locks_(max_elements),
        label_op_locks_(MAX_LABEL_OPERATION_LOCKS) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size();
    sq8distfunc_ = s->get_dist_func_sq8();
    quant_ = quant;
    dist_func_param_ = s->get_dist_func_param();
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;
    level_generator_ = r.level_generator_;
    update_probability_generator_ = r.update_probability_generator_;
    size_links_level0_ = r.size_links_level0_;
    size_data_per_element_ = r.size_data_per_element_;
    offsetData_ = r.offsetData_;
    label_offset_ = r.label_offset_;
    offsetLevel0_ = r.offsetLevel0_;

    data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    // data_level0_memory_ = r.data_level0_memory_;
    memcpy(data_level0_memory_, r.data_level0_memory_, max_elements_ * size_data_per_element_);
    // cur_element_count = r.cur_element_count;
    memcpy(&cur_element_count, &r.cur_element_count, sizeof(std::atomic<size_t>));
    
    visited_list_pool_ = new VisitedListPool(1, max_elements);

    enterpoint_node_ = r.enterpoint_node_;
    maxlevel_ = r.maxlevel_;
    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    memcpy(linkLists_, r.linkLists_, sizeof(void *) * max_elements_);
    element_levels_ = r.element_levels_;
    size_links_per_element_ = r.size_links_per_element_;
    for (size_t i = 0; i < cur_element_count; i++) {
      label_lookup_[getExternalLabel(i)] = i;
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      if (linkListSize == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        linkLists_[i] = (char *)malloc(linkListSize);
        memcpy(linkLists_[i], r.linkLists_[i], linkListSize);
      }
    }
    mult_ = r.mult_;
    revSize_ = r.revSize_;
  }
  HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                  SQ8Quantizer *quant = nullptr, size_t max_elements = 0)
   {
    loadIndex(location, s, quant, max_elements);
  }
  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M,
                  size_t ef_construction, SQ8Quantizer *quant,
                  size_t random_seed = 100)
      : link_list_locks_(max_elements),
        label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
        element_levels_(max_elements) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size();
    // fstdistfunc_ = s->get_dist_func();
    sq8distfunc_ = s->get_dist_func_sq8();
    quant_ = quant; // quant needs to be trained.
    dist_func_param_ = s->get_dist_func_param();
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ =
        size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);

    cur_element_count = 0;

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSW() {
    // free(data_level0_memory_);
    // for (tableint i = 0; i < cur_element_count; i++) {
    //   if (element_levels_[i] > 0)
    //     free(linkLists_[i]);
    // }
    // free(linkLists_);
    // delete visited_list_pool_;
  }

  struct CompareByFirst {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first < b.first;
    }
  };

  void setEf(size_t ef) { ef_ = ef; }

  inline std::mutex &getLabelOpMutex(labeltype label) const {
    // calculate hash
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    return label_op_locks_[lock_id];
  }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ +
            label_offset_),
           sizeof(labeltype));
    return return_label;
  }

  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
            label_offset_),
           &label, sizeof(labeltype));
  }

  inline labeltype *getExternalLabeLp(tableint internal_id) const {
    return (labeltype *)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + label_offset_);
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ +
            offsetData_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  size_t getMaxElements() { return max_elements_; }

  size_t getCurrentElementCount() { return cur_element_count; }

  size_t getDeletedCount() { return num_deleted_; }

  LPool<int, tableint>
  searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    // vl_type *visited_array = vl->mass;
    // vl_type visited_array_tag = vl->curV;
    LPool<int, tableint> pool(ef_construction_);
    dist_t dist = sq8distfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
    pool.emplace(dist, ep_id);
    // visited_array[ep_id] = visited_array_tag;
    vl->set(ep_id);
    while (!pool.empty()) {
      std::pair<int, tableint> curr_el_pair = pool.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);
      
      int *data;
      if (layer == 0) {
        data = (int *)get_linklist0(curNodeNum);
      } else {
        data = (int *)get_linklist(curNodeNum, layer);
      }
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      // _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      // _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
#ifdef USE_SSE
        // _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        // if (visited_array[candidate_id] == visited_array_tag)
        //   continue;
        // visited_array[candidate_id] = visited_array_tag;
        if (vl->get(candidate_id)) continue;
        vl->set(candidate_id);
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = sq8distfunc_(data_point, currObj1, dist_func_param_);
        pool.emplace(dist1, candidate_id);
      }
    }
    visited_list_pool_->releaseVisitedList(vl);
    pool.reset();
    return std::move(pool);
  }
  const size_t B1[4] = {1500'000,  2000'000, 3000'000, 4000'000};
  template <bool has_deletions, bool collect_metrics = false>
  PriorityQueue<int, tableint>
  searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                    BaseFilterFunctor *isIdAllowed = nullptr) const {
    ef = 1000;
    if (cur_element_count > B1[0]) ef = 1100;
    if (cur_element_count > B1[1]) ef = 1500;
    if (cur_element_count > B1[2]) ef = 1800;
    if (cur_element_count > B1[2]) ef = 1800;
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    // vl_type *visited_array = vl->mass;
    // vl_type visited_array_tag = vl->curV;
    // std::priority_queue<std::pair<dist_t, tableint>,
    //                     std::vector<std::pair<dist_t, tableint>>,
    //                     CompareByFirst>
    //     top_candidates;
    // std::priority_queue<std::pair<dist_t, tableint>,
    //                     std::vector<std::pair<dist_t, tableint>>,
    //                     CompareByFirst>
    //     candidate_set;
    PriorityQueue<int, tableint>  top_candidates(ef);
    PriorityQueue<int, tableint>  candidate_set{};
    dist_t lowerBound;
//    if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
//        ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
      dist_t dist = sq8distfunc_(data_point, getDataByInternalId(ep_id),
                                 dist_func_param_);
      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
      candidate_set.emplace(-dist, ep_id);
//      cout <<"1111" << endl;
//    } else {
//      lowerBound = std::numeric_limits<dist_t>::max();
//      candidate_set.emplace(-lowerBound, ep_id);
//      cout << "222222" << endl;
//    }

    // visited_array[ep_id] = visited_array_tag;
    vl->set(ep_id);
    int num = 0;

    while (!candidate_set.empty()) {
      std::pair<int, tableint> current_node_pair = candidate_set.top();

      if ((-current_node_pair.first) > lowerBound) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      int *data = (int *)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);
      //                bool cur_node_deleted =
      //                isMarkedDeleted(current_node_id);
//      if (collect_metrics) {
//        metric_hops++;
//        metric_distance_computations += size;
//      }

#ifdef USE_SSE
      // _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      // _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ +
                       (*(data + 1)) * size_data_per_element_ + offsetData_,
                   _MM_HINT_T0);
      _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
        num++;
#ifdef USE_SSE
        // _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ +
                         (*(data + j + 1)) * size_data_per_element_ +
                         offsetData_,
                     _MM_HINT_T0); ////////////
#endif
        // if (!(visited_array[candidate_id] == visited_array_tag)) {
        //   visited_array[candidate_id] = visited_array_tag;
        if (!vl->get(candidate_id)) {
          vl->set(candidate_id);
          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t dist = sq8distfunc_(data_point, currObj1, dist_func_param_);

          if (top_candidates.size() < ef || lowerBound > dist) {
            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
            _mm_prefetch(data_level0_memory_ +
                             candidate_set.top().second *
                                 size_data_per_element_ +
                             offsetLevel0_, ///////////
                         _MM_HINT_T0);      ////////////////////////
#endif

//            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
//                ((!isIdAllowed) ||
//                 (*isIdAllowed)(getExternalLabel(candidate_id))))
              top_candidates.emplace(dist, candidate_id);

            if (top_candidates.size() > ef)
              top_candidates.pop();

            if (!top_candidates.empty())
              lowerBound = top_candidates.top().first;
          }
        }
      }
    }
//    cout << "num = " << num << endl;
    visited_list_pool_->releaseVisitedList(vl);
    return std::move(top_candidates);
  }
    template <bool has_deletions, bool collect_metrics = false>
    PriorityQueue<int, tableint>
    searchBaseLayerST_filter(tableint ep_id, const void *data_point, size_t ef,
                             uint32_t query_type, int32_t c, float l, float r, int sel, 
                             BaseFilterFunctor *isIdAllowed = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        // vl_type *visited_array = vl->mass;
        // vl_type visited_array_tag = vl->curV;
        int vis_time_threshold = 0;
        float rate = sel / (float)cur_element_count;
        if (query_type == 2) {
          ef = 1900 - rate * 500;
          vis_time_threshold = 52;
        } else {
          ef = 1400 - rate * 500;
          vis_time_threshold = 32 + 4 - 4 * rate;
        }
        PriorityQueue<int, tableint> top_candidates(ef);
        PriorityQueue<int, tableint> candidate_set{};
        dist_t lowerBound = std::numeric_limits<int>::max();
        // todo: type2 和 3 的进入点选择，方式及数量
        // int lower = std::lower_bound(time_list.begin(), time_list.end(), l) - time_list.begin();
        // int upper = std::upper_bound(time_list.begin(), time_list.end(), r) - time_list.begin() - 1;
        // int sel_num = 10;
        // int len = (upper - lower + 1) / sel_num;
        // // PriorityQueue<dist_t, tableint> sample_pq(3);
        // for (int i = lower; i <= upper; i += len) {
        //   auto iter = label_lookup_.find(i);
        //   tableint ep_id = iter->second;
        //   dist_t dist = sq8distfunc_(data_point, getDataByInternalId(ep_id),
        //                                dist_func_param_);
        //   // sample_pq.emplace(dist, ep_id);
        //   candidate_set.emplace(-dist, ep_id);
        // }
        // // while (!sample_pq.empty()) {
        // //   top_candidates.push(sample_pq.top());
        // //   candidate_set.push(std::make_pair(-sample_pq.top().first, sample_pq.top().second));
        // //   sample_pq.pop();
        // // }
        // if (time_list[getExternalLabel(ep_id)] >= l && time_list[getExternalLabel(ep_id)] <= r) {
        //   // if ((ep_id - lower) % len != 0) {
        //     dist_t dist = sq8distfunc_(data_point, getDataByInternalId(ep_id),
        //                               dist_func_param_);
        //     top_candidates.emplace(dist, ep_id);
        //     candidate_set.emplace(-dist, ep_id);
        //   // }
        // } else {
        //   ep_id = enterpoint_node_;
        //   candidate_set.emplace(-lowerBound, ep_id);
        // }
        // lowerBound = top_candidates.top().first;
        
        if (time_list[getExternalLabel(ep_id)] >= l && time_list[getExternalLabel(ep_id)] <= r) {
            dist_t dist = sq8distfunc_(data_point, getDataByInternalId(ep_id),
                                       dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            ep_id = enterpoint_node_;
            lowerBound = std::numeric_limits<int>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        // visited_array[ep_id] = visited_array_tag;
        vl->set(ep_id);
        int num1 = 0, num2 = 0;

        while (!candidate_set.empty()) {
            std::pair<int, tableint> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound &&
                (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                break;
            }
            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;
            // if (visited_array[current_node_id] == visited_array_tag) {
            //   continue;
            // }
            const std::pair<float, tableint> *c_edge = edges.data() + current_node_id * (maxM0_ + 1);
            int c_size = c_edge[0].second;
// todo： 测试预取的影响
//#ifdef USE_SSE
//            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
//            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
//            _mm_prefetch(data_level0_memory_ +
//                         (*(data + 1)) * size_data_per_element_ + offsetData_,
//                         _MM_HINT_T0);
//            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
//#endif

            int vis_time = 0;

            for (int k = 1; k <= c_size; k++) {
                auto edge = *(c_edge + k);
                const std::pair<float, tableint> *n_edge = edges.data() + edge.second * (maxM0_ + 1);
                // _mm_prefetch((char *)(n_edge + 1), _MM_HINT_T0);
                int n_size = n_edge[0].second;
                int candidate_id = edge.second;
// #ifdef USE_SSE
//                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
//                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                _mm_prefetch(data_level0_memory_ +
//                             (*(data + 1)) * size_data_per_element_ + offsetData_,
//                             _MM_HINT_T0);
//                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
// #endif
                
                for (int j = 1; j <= n_size; j++) {
                    if(vis_time >= vis_time_threshold) break;
                    auto edge = *(n_edge + j);
                    // _mm_prefetch((char *)(n_edge + j + 1), _MM_HINT_T0);
                    if (edge.first < l) continue;
                    if (edge.first > r) continue;
                    int candidate_id = edge.second;
                    vis_time++;
                    num2++;
// #ifdef USE_SSE
//                   //  _mm_prefetch((char *)(visited_array + *(tmp_data + j + 1)), _MM_HINT_T0);
//                    _mm_prefetch(data_level0_memory_ +
//                                 (*(tmp_data + j + 1)) * size_data_per_element_ +
//                                 offsetData_,
//                                 _MM_HINT_T0); ////////////
// #endif
                    // if (!(visited_array[candidate_id] == visited_array_tag)) {
                    //     visited_array[candidate_id] = visited_array_tag;
                    if (!vl->get(candidate_id)) {
                        vl->set(candidate_id);
                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = sq8distfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
//#ifdef USE_SSE
//                            _mm_prefetch(data_level0_memory_ +
//                                         candidate_set.top().second *
//                                         size_data_per_element_ +
//                                         offsetLevel0_, ///////////
//                                         _MM_HINT_T0);      ////////////////////////
//#endif

//                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
//                                ((!isIdAllowed) ||
//                                 (*isIdAllowed)(getExternalLabel(candidate_id))))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
              if (edge.first < l || edge.first > r) continue;
              vis_time++;
              num1++;

//#ifdef USE_SSE
//                _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
//                _mm_prefetch(data_level0_memory_ +
//                             (*(data + j + 1)) * size_data_per_element_ +
//                             offsetData_,
//                             _MM_HINT_T0); ////////////
//#endif
                // if (!(visited_array[candidate_id] == visited_array_tag)) {
                //     visited_array[candidate_id] = visited_array_tag;
                  if (!vl->get(candidate_id)) {
                    vl->set(candidate_id);
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = sq8distfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
//#ifdef USE_SSE
//                        _mm_prefetch(data_level0_memory_ +
//                                     candidate_set.top().second *
//                                     size_data_per_element_ +
//                                     offsetLevel0_, ///////////
//                                     _MM_HINT_T0);      ////////////////////////
//#endif

//                        if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
//                            ((!isIdAllowed) ||
//                             (*isIdAllowed)(getExternalLabel(candidate_id))))
                            top_candidates.emplace(dist, candidate_id);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

        }
//        cout << "num1 = " << num1 << ", num2 = " << num2 << endl;
        visited_list_pool_->releaseVisitedList(vl);
        return std::move(top_candidates);
    }
  void getNeighborsByHeuristic2(
      LPool<int, tableint> &pool,
      const size_t M) {
    if (pool.size() < M) {
      return;
    }
    LPool<int, tableint> res(pool.capacity_);
    for (int i = 0; i < pool.size(); i++) {
      if (res.size() >= M) break;
      dist_t dist_to_query = pool.data_[i].first;
      auto data = getDataByInternalId(pool.id(i));
      bool good = true;
      for (int j = 0; j < res.size(); j++) {
        dist_t curdist = sq8distfunc_(getDataByInternalId(res.id(j)), 
                                      data,
                                      dist_func_param_);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        res.push(pool.data_[i]);
      }
    }
    res.reset();
    pool = std::move(res);
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist0(tableint internal_id,
                                 char *data_level0_memory_) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  }

  linklistsizeint *get_linklist_at_level(tableint internal_id,
                                         int level) const {
    return level == 0 ? get_linklist0(internal_id)
                      : get_linklist(internal_id, level);
  }

  tableint mutuallyConnectNewElement(
      const void *, tableint cur_c,
      LPool<int, tableint> &pool,
      int level, bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(pool, M_);

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    for (int i = 0; i < pool.size(); i++) {
      selectedNeighbors.push_back(pool.id(i));
    }
    tableint next_closest_entry_point = selectedNeighbors[0];
    {
      // lock only during the update
      // because during the addition the lock for cur_c is already acquired
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_c],
                                        std::defer_lock);
      linklistsizeint *ll_cur;
      if (level == 0)
        ll_cur = get_linklist0(cur_c);
      else
        ll_cur = get_linklist(cur_c, level);

      setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(
          link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = getListCount(ll_other);

      tableint *data = (tableint *)(ll_other + 1);

      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or run
      // the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);
        } else {
          // finding the "weakest" element to replace it with the new one
          int d_max = sq8distfunc_(
              getDataByInternalId(cur_c),
              getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
          LPool<int, tableint> candidates(sz_link_list_other);
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                sq8distfunc_(getDataByInternalId(data[j]),
                             getDataByInternalId(selectedNeighbors[idx]),
                             dist_func_param_),
                data[j]);
          }
          getNeighborsByHeuristic2(candidates, Mcurmax);
          for (int i = 0; i < candidates.size(); i++) {
            data[i] = candidates.id(i);
          }
          setListCount(ll_other, candidates.size());
        }
      }
    }

    return next_closest_entry_point;
  }

  void resizeIndex(size_t new_max_elements) {

    delete visited_list_pool_;
    visited_list_pool_ = new VisitedListPool(1, new_max_elements);

    element_levels_.resize(new_max_elements);

    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

    // Reallocate base layer
    char *data_level0_memory_new = (char *)realloc(
        data_level0_memory_, new_max_elements * size_data_per_element_);
    data_level0_memory_ = data_level0_memory_new;

    // Reallocate all other layers
    char **linkLists_new =
        (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_,
                 cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize =
          element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i]
                                 : 0;
      writeBinaryPOD(output, linkListSize);
      if (linkListSize)
        output.write(linkLists_[i], linkListSize);
    }
    output.close();
  }
  
  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                 SQ8Quantizer *quant = nullptr, size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count)
      max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    // fstdistfunc_ = s->get_dist_func();
    sq8distfunc_ = s->get_dist_func_sq8();
    quant_ = quant;
    dist_func_param_ = s->get_dist_func_param();
    auto pos = input.tellg();

    /// Optional - check if index is ok:
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {

      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize != 0) {
        input.seekg(linkListSize, input.cur);
      }
    }

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
      label_lookup_[getExternalLabel(i)] = i;
      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        element_levels_[i] = linkListSize / size_links_per_element_;
        linkLists_[i] = (char *)malloc(linkListSize);
        input.read(linkLists_[i], linkListSize);
      }
    }

    for (size_t i = 0; i < cur_element_count; i++) {
      if (isMarkedDeleted(i)) {
        num_deleted_ += 1;
        if (allow_replace_deleted_)
          deleted_elements.insert(i);
      }
    }

    input.close();

    return;
  }

  template <typename data_t>
  std::vector<data_t> getDataByLabel(labeltype label) const {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;
    lock_table.unlock();

    char *data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t *)dist_func_param_);
    std::vector<data_t> data;
    data_t *data_ptr = (data_t *)data_ptrv;
    for (int i = 0; i < (int)dim; i++) {
      data.push_back(*data_ptr);
      data_ptr += 1;
    }
    return data;
  }

  /*
   * Marks an element with the given label deleted, does NOT really change the
   * current graph.
   */
  void markDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;
    lock_table.unlock();

    markDeletedInternal(internalId);
  }

  /*
   * Uses the last 16 bits of the memory for the linked list size to store the
   * mark, whereas maxM0_ has to be limited to the lower 16 bits, however, still
   * large enough in almost all cases.
   */
  void markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (!isMarkedDeleted(internalId)) {
      unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
      *ll_cur |= DELETE_MARK;
      num_deleted_ += 1;
      if (allow_replace_deleted_) {
        std::unique_lock<std::mutex> lock_deleted_elements(
            deleted_elements_lock);
        deleted_elements.insert(internalId);
      }
    }
  }

  /*
   * Removes the deleted mark of the node, does NOT really change the current
   * graph.
   *
   * Note: the method is not safe to use when replacement of deleted elements is
   * enabled, because elements marked as deleted can be completely removed by
   * addPoint
   */
  void unmarkDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    tableint internalId = search->second;
    lock_table.unlock();

    unmarkDeletedInternal(internalId);
  }

  /*
   * Remove the deleted mark of the node.
   */
  void unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (isMarkedDeleted(internalId)) {
      unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
      *ll_cur &= ~DELETE_MARK;
      num_deleted_ -= 1;
      if (allow_replace_deleted_) {
        std::unique_lock<std::mutex> lock_deleted_elements(
            deleted_elements_lock);
        deleted_elements.erase(internalId);
      }
    }
  }

  /*
   * Checks the first 16 bits of the memory to see if the element is marked
   * deleted.
   */
  bool isMarkedDeleted(tableint internalId) const {
    unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
    return *ll_cur & DELETE_MARK;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  /*
   * Adds point. Updates the point if it is already in the index.
   * If replacement of deleted elements is enabled: replaces previously deleted
   * point if any, updating it with new point
   */
  void addPoint(const void *data_point, labeltype label,
                bool replace_deleted = false) {
    data_point = quant_->get_data(label);
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
    addPoint(data_point, label, -1);
  }

  void updatePoint(const void *dataPoint, tableint internalId,
                   float updateNeighborProbability) {
  }

  void repairConnectionsForUpdate(const void *dataPoint,
                                  tableint entryPointInternalId,
                                  tableint dataPointInternalId,
                                  int dataPointLevel, int maxLevel) {
    
  }

  std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
    std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
    unsigned int *data = get_linklist_at_level(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint *ll = (tableint *)(data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
  }

  tableint addPoint(const void *data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
      // Checking if the element with the same label already exists
      // if so, updating it *instead* of creating a new element.
      std::unique_lock<std::mutex> lock_table(label_lookup_lock);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        tableint existingInternalId = search->second;
        lock_table.unlock();

//        if (isMarkedDeleted(existingInternalId)) {
//          unmarkDeletedInternal(existingInternalId);
//        }
        updatePoint(data_point, existingInternalId, 1.0);

        return existingInternalId;
      }

      cur_c = cur_element_count;
      cur_element_count++;
      label_lookup_[label] = cur_c;
    }
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0)
      curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
      templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
           0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    if (curlevel) {
      linkLists_[cur_c] =
          (char *)malloc(size_links_per_element_ * curlevel + 1);
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }
    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist = sq8distfunc_(data_point, getDataByInternalId(currObj),
                                      dist_func_param_);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              dist_t d = sq8distfunc_(data_point, getDataByInternalId(cand),
                                      dist_func_param_);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

//      bool epDeleted = isMarkedDeleted(enterpoint_copy);
      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {

        LPool<int, tableint>
            top_candidates = searchBaseLayer(currObj, data_point, level);
        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                            level, false);
      }
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }
    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
    return cur_c;
  }

  LPool<dist_t, labeltype>
  searchKnn(const void *query_data, size_t k,
            BaseFilterFunctor *isIdAllowed = nullptr) const {
    return LPool<dist_t, labeltype>(k);
  }
  static constexpr int nef[] = {2200, 1500, 1500, 1200};
    PriorityQueue<int, labeltype> 
    searchKnn_filter(const void *query_data, size_t k,
              uint32_t query_type, int32_t c, float l, float r, int sel,
                     BaseFilterFunctor *isIdAllowed = nullptr) {
        PriorityQueue<int, labeltype> result(k);

        tableint currObj = enterpoint_node_;
        dist_t curdist = sq8distfunc_(
                query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);


        PriorityQueue<int, tableint> top_candidates(std::max(ef_, k));

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = sq8distfunc_(query_data, getDataByInternalId(cand),
                                            dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        if(query_type == 0 || query_type == 1) {
            top_candidates = searchBaseLayerST<false, false>(
                    currObj, query_data, k, isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST_filter<false, false>(
                    currObj, query_data, k, query_type, c, l, r, sel, isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<int, tableint> rez = top_candidates.top();
            result.push(std::pair<int, labeltype>(rez.first,
                                                     getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return std::move(result);
    }

  void checkIntegrity() {
    int connections_checked = 0;
    std::vector<int> inbound_connections_num(cur_element_count, 0);
    for (int i = 0; i < cur_element_count; i++) {
      for (int l = 0; l <= element_levels_[i]; l++) {
        linklistsizeint *ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        std::unordered_set<tableint> s;
        for (int j = 0; j < size; j++) {
          assert(data[j] > 0);
          assert(data[j] < cur_element_count);
          assert(data[j] != i);
          inbound_connections_num[data[j]]++;
          s.insert(data[j]);
          connections_checked++;
        }
        assert(s.size() == size);
      }
    }
    if (cur_element_count > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count; i++) {
        assert(inbound_connections_num[i] > 0);
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked
              << " connections\n";
  }
};
} // namespace hnswlib