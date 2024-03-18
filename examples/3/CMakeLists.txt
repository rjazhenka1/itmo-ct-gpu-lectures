cmake_minimum_required(VERSION 3.2.2)
project(ocl3 C)

set(CL_TARGET_OPENCL_VERSION "100")

find_package(OpenCL REQUIRED)
add_executable(ocl3 main.c)
target_include_directories(ocl3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ocl3 OpenCL::OpenCL)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")