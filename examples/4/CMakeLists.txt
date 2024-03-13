cmake_minimum_required(VERSION 3.2.2)
project(ocl4 C)

set(CL_TARGET_OPENCL_VERSION "100")

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(ocl4 main.c)
target_include_directories(ocl4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ocl4 OpenCL::OpenCL OpenMP::OpenMP_C)