cmake_minimum_required(VERSION 3.5)

project(csrt-tracker LANGUAGES CXX CUDA)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
