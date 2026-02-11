cmake_minimum_required(VERSION 3.25)
project(event_horizon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(event_horizon main.cpp)

target_compile_options(event_horizon PRIVATE
    -fno-exceptions
    -fno-rtti
    -Wall -Wextra -Wpedantic
    -O2
    -march=native
)

target_link_libraries(event_horizon PRIVATE pthread)
