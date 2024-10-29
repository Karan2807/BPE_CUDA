cmake_minimum_required(VERSION 3.20)
project(bpe-cuda LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(CUDAToolkit)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Find Boost package
find_package(Boost 1.40 REQUIRED COMPONENTS regex) # Specify the minimum version you need

# Specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)
# Include directories

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0") # no optimizations


include_directories(.)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(bpe-cuda
        src/headers/bpe_seq.h
        src/CommonData.cpp
        src/headers/CommonData.h
        src/train_and_tokenize.cpp
        src/detokenize.cpp
        src/train_tokenizer.cu
        src/bpe_main.cpp
        src/main.cpp
        src/kernels.cu
        src/headers/cuda_bpe.h
        src/cuda_wrapper.cpp
)
set_property(TARGET bpe-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bpe-cuda PRIVATE cudart PUBLIC ${Boost_LIBRARIES})