/**
 * This part is adapted from https://github.com/zilliztech/pyglass/blob/master/glass/neighbor.hpp.
 */
#pragma once
#include <algorithm>
#include "utils.h"

template <typename dist_t, typename id_t>
struct PriorityQueue {
  explicit PriorityQueue() : capacity(-1) {}
  explicit PriorityQueue(int capacity) : capacity(capacity), pool(capacity) {}
  PriorityQueue(PriorityQueue<dist_t, id_t> &&r) : capacity(r.capacity), sz(r.sz), pool(std::move(r.pool)) {}
  PriorityQueue(const PriorityQueue<dist_t, id_t> &r) : capacity(r.capacity), sz(r.sz), pool(r.pool) {}
  PriorityQueue operator = (PriorityQueue<dist_t, id_t> &&r) {
    capacity = r.capacity;
    sz = r.sz;
    pool = std::move(r.pool);
    return *this;
  }
  PriorityQueue operator = (const PriorityQueue<dist_t, id_t> &r) {
    capacity = r.capacity;
    sz = r.sz;
    pool = r.pool;
    return *this;
  }
  void emplace(dist_t dist, id_t id) {
    if (capacity != -1) {
      if (sz < capacity) {
        pool[sz] = std::make_pair(dist, id);
        std::push_heap(pool.begin(), pool.begin() + ++sz);
      } else if (dist < pool[0].first) {
        sift_down(0, id, dist);
      }
    } else {
      pool.push_back(std::make_pair(dist, id));
      std::push_heap(pool.begin(), pool.begin() + ++sz);
    }
    // pool.push_back(std::make_pair(dist, id));
    // std::push_heap(pool.begin(), pool.begin() + ++sz);
  }
  void push(const std::pair<dist_t, id_t> &p) {
    emplace(p.first, p.second);
  }
  void pop() {
    std::pop_heap(pool.begin(), pool.begin() + sz--);
    pool.pop_back();
  }
  void build() {
    std::make_heap(pool.begin(), pool.end() + sz);
  }
  int size() const { return sz; }
  bool empty() const { return sz == 0; }
  std::pair<dist_t, id_t> top() const { return *pool.begin(); }
  void sift_down(int i, id_t id, dist_t dist) {
    pool[0] = std::make_pair(dist, id);
    for (; 2 * i + 1 < sz;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].first > dist) {
        j = l;
      }
      if (r < sz && pool[r].first > std::max(pool[l].first, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = std::make_pair(dist, id);
  }
  int sz = 0, capacity;
  std::vector<std::pair<dist_t, id_t>> pool;
};
template <typename dist_t, typename id_t> struct LPool {
  explicit LPool(int capacity) : capacity_(capacity), data_(capacity_ + 1) {}
  LPool(LPool<dist_t, id_t> &&r) : size_(r.size_), cur_(r.cur_), capacity_(r.capacity_), data_(std::move(r.data_)) {}
  LPool(const LPool<dist_t, id_t> &r) : size_(r.size_), cur_(r.cur_), capacity_(r.capacity_), data_(r.data_) {}
  LPool operator = (LPool<dist_t, id_t> &&r) {
    size_ = r.size_;
    cur_ = r.cur_;
    capacity_ = r.capacity_;
    data_ = std::move(r.data_);
    return *this;
  }
  LPool operator = (const LPool<dist_t, id_t> &r) {
    size_ = r.size_;
    cur_ = r.cur_;
    capacity_ = r.capacity_;
    data_ = r.data_;
    return *this;
  }
  int find_bsearch(dist_t dist) {
    // if (size_ && dist >= data_[size_ - 1].first) {
    //   return size_;
    // }
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].first > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  bool emplace(dist_t dist, id_t id) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].first) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(std::pair<dist_t, id_t>));
    data_[lo] = std::make_pair(dist, id);
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }
  bool push(const std::pair<dist_t, id_t> &p) {
    return emplace(p.first, p.second);
  }
  std::pair<dist_t, id_t> pop() {
    set_checked(data_[cur_].second);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].second)) {
      cur_++;
    }
    return std::make_pair(data_[pre].first, get_id(data_[pre].second));
  }
  void reset(int new_size) {
    cur_ = 0;
    size_ = new_size;
    for (int i = 0; i < size_; i++) {
      data_[i].second = get_id(data_[i].second);
    }
  }
  void reset() {
    cur_ = 0;
    for (int i = 0; i < size_; i++) {
      data_[i].second = get_id(data_[i].second);
    }
  }
  bool empty() const { return cur_ >= size_; }
  id_t id(int i) const { return get_id(data_[i].second); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }
  const id_t kMask = 2147483647;
  id_t get_id(int id) const { return id & kMask; }
  void set_checked(id_t &id) { id |= 1 << 31; }
  bool is_checked(id_t id) { return id >> 31 & 1; }

  int size_ = 0, cur_ = 0, capacity_;
  std::vector<std::pair<dist_t, id_t>, align_alloc<std::pair<dist_t, id_t>>> data_;
};