//This will serve as a binding file for the functions we have defined in mysampling.h
//and mysampling.cc file
#include <pybind11/pybind11.h>
#include "wsample/mysampling.h"
#include <pybind11/stl.h>
//for data structure between cpp and torch
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(sampling_module, m) {
  m.def("sample", &sample<false, true>);
  m.def("test", &test);
  m.def("weighted_sample", &weighted_sample);
  m.def("weighted_random_walk", &weighted_random_walk);
  //m.def("sample1", &sample1<false,true>);
}

