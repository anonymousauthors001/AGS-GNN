set(HEADER_LIST
    "${PROJECT_SOURCE_DIR}/include/utils.h"
    "${PROJECT_SOURCE_DIR}/include/mysamplings.h"
)

#set(LIBTORCH_PY "/home/sferdou/.conda/envs/cent7/2020.11-py38/py310cu117pyg200/lib/python3.10/site-packages/torch/lib/libtorch_python.so")
set(LIBTORCH_PY "/home/das90/.conda/envs/cent7/2020.11-py38/py38cu11/lib/python3.8/site-packages/torch/lib/libtorch_python.so")


#Create the library which we will bind to python
add_library(wsample SHARED mysampling.cc)
target_link_libraries(wsample PUBLIC ${TORCH_LIBRARIES})
target_include_directories(wsample PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(wsample PUBLIC cxx_std_14)
target_compile_options(wsample PRIVATE -O3)

pybind11_add_module(sampling_module module.cc)
target_link_libraries(sampling_module PRIVATE pybind11::module wsample ${TORCH_LIBRARIES})
#This is important
target_link_libraries(sampling_module PRIVATE ${LIBTORCH_PY})
target_include_directories(sampling_module PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(sampling_module PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(sampling_module PRIVATE cxx_std_14)

# Debug info
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

