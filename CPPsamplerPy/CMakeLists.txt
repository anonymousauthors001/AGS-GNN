cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Sampling)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Set the path to the installed PyTorch library for cmake config(adjust as needed)
#set(Torch_DIR "/home/sferdou/.conda/envs/cent7/2020.11-py38/py310cu117pyg200/lib/python3.10/site-packages/torch/share/Torch/cmake")
set(Torch_DIR "/home/das90/.conda/envs/cent7/2020.11-py38/py38cu11/lib/python3.8/site-packages/torch/share/cmake/Torch")
#/home/das90/.conda/envs/cent7/2020.11-py38/py38cu11/lib/python3.8/site-packages/torch/share/Torch/cmake

# Set the path to the installed Pybind11 library for cmake config(adjust as needed)
set(pybind11_DIR "/home/das90/.conda/envs/cent7/2020.11-py38/py38cu11/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

# Locate pyTorch
find_package(Torch REQUIRED)
# Locate pybind11
find_package(pybind11 REQUIRED)

add_subdirectory(src)
add_subdirectory(apps)


