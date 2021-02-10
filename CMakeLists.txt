cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)
#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES})
