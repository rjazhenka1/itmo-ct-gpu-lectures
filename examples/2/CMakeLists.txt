cmake_minimum_required(VERSION 3.2.2)
project(ocl2 C)

set(CL_TARGET_OPENCL_VERSION "100")

find_package(OpenCL REQUIRED)
add_executable(ocl2 main.c)
target_include_directories(ocl2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ocl2 OpenCL::OpenCL)