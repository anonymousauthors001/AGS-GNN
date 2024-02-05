
#include "wsample/mysampling.h"
#include <iostream>
#include <string>
//#include <torch/torch.h>
//#include "parallel_hashmap/phmap.h"
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#include <process.h>
#endif

void test() 
{

  std::cout<<"hi world!"<<std::endl;
}

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
//Must have to instantiate the template function. Otherwise there will be linking 
//error. Instantiation is creating a instant of templated function for example here we
//use false and true for the two bool value in the template definition
template tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample<false,true>(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors);


tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
weighted_sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors, 
       const torch::Tensor &weights, const bool replace, const bool directed) {


  //cout << "weights: " << weights << endl;
  //cout << "replace: " <<replace<<endl;
  //cout << "directed: "<< directed<<endl;

  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  phmap::flat_hash_map<int64_t, int64_t> to_local_node;

  auto *colptr_data = colptr.data_ptr<int64_t>();
  auto *row_data = row.data_ptr<int64_t>();
  auto *input_node_data = input_node.data_ptr<int64_t>();
  //auto *weights_data = weights.data_ptr<int64_t>();

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
      } 
      else{

        const auto sel_col_weights = weights.slice(/*dim=*/0, col_start, col_end);
        //cout<<"Col weights"<<sel_col_weights<<endl;

        double sum = sel_col_weights.sum().item<double>();
        const auto col_weights = sel_col_weights/sum;
        //cout<<"Col norm weights"<<col_weights<<endl;
        //cout<<"Col weights value: "<<col_weights.value()<<endl;


        torch::Tensor r_samples = choice(col_count, num_samples, replace, col_weights);
        auto *r_samples_data = r_samples.data_ptr<int64_t>();

        /*
        for (int64_t j = 0; j < num_samples; j++) {
          auto &test = r_samples_data[j];
          cout<<"r_samples: "<< test<<endl;
        }*/

        for (int64_t j = 0; j < num_samples; j++) {
          auto &choice_offset = r_samples_data[j];
          const int64_t offset = col_start + choice_offset;
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


torch::Tensor weighted_random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start, int64_t walk_length, torch::Tensor weights) {

  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);
  CHECK_CPU(weights);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);
  CHECK_INPUT(weights.dim() == 1);

  // auto rand = torch::rand({start.size(0), walk_length}, start.options().dtype(torch::kFloat));

  auto L = walk_length + 1;
  auto out = torch::full({start.size(0), L}, -1, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  // auto rand_data = rand.data_ptr<float>();
  auto out_data = out.data_ptr<int64_t>();

  for (auto n = 0; n < start.size(0); n++) {
    auto cur = start_data[n];
    out_data[n * L] = cur;

    int64_t row_start, row_end;
    for (auto l = 0; l < walk_length; l++) {
      row_start = rowptr_data[cur];
      row_end = rowptr_data[cur + 1];

      auto row_count = row_end - row_start;
      if (row_count<1){
        out_data[n * L + l + 1] = cur;
        continue;
      }

      auto col_probs = weights.slice(/*dim=*/0, row_start, row_end);
      col_probs = col_probs.clamp_min_(0.0001);
      col_probs = col_probs/col_probs.sum();
      
      //cout<<"col probs: "<<col_probs<<endl;

      auto r_samples = choice(row_count, 1, true, col_probs);
      //cout<<"r sample "<<r_samples[0];

      cur = col_data[row_start + r_samples[0].item<int64_t>()];

      //cur = col_data[row_start + int64_t(rand_data[n * walk_length + l] *(row_end - row_start))];
      out_data[n * L + l + 1] = cur;
    }
  }

  return out;
}



