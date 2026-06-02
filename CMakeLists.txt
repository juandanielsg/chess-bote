cmake_minimum_required(VERSION 3.15...3.28)
project(chess_bote LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC src/cpp/core.cpp)

target_compile_options(_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -funroll-loops>)

# Land the compiled module inside the Python package.
install(TARGETS _core LIBRARY DESTINATION chess_bote)
