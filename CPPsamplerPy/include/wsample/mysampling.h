#pragma once

#include "wsample/utils.h"
#include <vector>
#include <tuple>
#include <iostream>
#include <string>
//#include <torch/torch.h>
//#include "parallel_hashmap/phmap.h"
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#include <process.h>
#endif

using namespace std;

typedef phmap::flat_hash_map<pair<int64_t, int64_t>, int64_t> temporarl_edge_dict;

template <bool replace, bool directed>
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors); 

void test();

template <bool replace, bool directed>
void sample1( torch::Tensor &colptr,  torch::Tensor &row,
        torch::Tensor &input_node); 


tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
weighted_sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors, 
       const torch::Tensor &weights, const bool replace, const bool directed);


torch::Tensor weighted_random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start, int64_t walk_length, torch::Tensor weights);

/*
template <bool replace, bool directed>
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors) {

  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  phmap::flat_hash_map<int64_t, int64_t> to_local_node;

  auto *colptr_data = colptr.data_ptr<int64_t>();
  auto *row_data = row.data_ptr<int64_t>();
  auto *input_node_data = input_node.data_ptr<int64_t>();

  for (int64_t i = 0; i < input_node.numel(); i++) {
    const auto &v = input_node_data[i];
    samples.push_back(v);
    to_local_node.insert({v, i});
  }

  vector<int64_t> rows, cols, edges;

  int64_t begin = 0, end = samples.size();
  for (int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++) {
    const auto &num_samples = num_neighbors[ell];
    for (int64_t i = begin; i < end; i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      const auto col_count = col_end - col_start;

      if (col_count == 0)
        continue;

      if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else if (replace) {
        for (int64_t j = 0; j < num_samples; j++) {
          const int64_t offset = col_start + uniform_randint(col_count);
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else {
        unordered_set<int64_t> rnd_indices;
        for (int64_t j = col_count - num_samples; j < col_count; j++) {
          int64_t rnd = uniform_randint(j);
          if (!rnd_indices.insert(rnd).second) {
            rnd = j;
            rnd_indices.insert(j);
          }
          const int64_t offset = col_start + rnd;
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      }
    }
    begin = end, end = samples.size();
  }

  if (!directed) {
    phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
    for (int64_t i = 0; i < (int64_t)samples.size(); i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      for (int64_t offset = col_start; offset < col_end; offset++) {
        const auto &v = row_data[offset];
        iter = to_local_node.find(v);
        if (iter != to_local_node.end()) {
          rows.push_back(iter->second);
          cols.push_back(i);
          edges.push_back(offset);
        }
      }
    }
  }

  return make_tuple(from_vector<int64_t>(samples), from_vector<int64_t>(rows),
                    from_vector<int64_t>(cols), from_vector<int64_t>(edges));
}
*/
