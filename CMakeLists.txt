cmake_minimum_required(VERSION 3.16)
project(MiniXGBoost VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(minixgb src/minixgb.cpp)
target_include_directories(minixgb PUBLIC include)

if(MSVC)
    target_compile_options(minixgb PRIVATE /W4 /permissive-)
else()
    target_compile_options(minixgb PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(minixgb_demo src/main.cpp)
target_link_libraries(minixgb_demo PRIVATE minixgb)

enable_testing()
add_executable(minixgb_tests tests/test_minixgb.cpp)
target_link_libraries(minixgb_tests PRIVATE minixgb)
add_test(NAME minixgb_tests COMMAND minixgb_tests)

