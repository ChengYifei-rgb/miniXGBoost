cmake_minimum_required(VERSION 3.16)
project(MiniXGBoost VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MINIXGB_BUILD_DEMO "Build the MiniXGBoost command-line demo" ON)
option(MINIXGB_BUILD_TESTS "Build the MiniXGBoost test suite" ON)

add_library(minixgb src/minixgb.cpp)
add_library(MiniXGBoost::minixgb ALIAS minixgb)

target_compile_features(minixgb PUBLIC cxx_std_17)
target_include_directories(minixgb
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MSVC)
    target_compile_options(minixgb PRIVATE /W4 /permissive-)
else()
    target_compile_options(minixgb PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MINIXGB_BUILD_DEMO)
    add_executable(minixgb_demo src/main.cpp)
    target_link_libraries(minixgb_demo PRIVATE MiniXGBoost::minixgb)
endif()

include(CTest)
if(BUILD_TESTING AND MINIXGB_BUILD_TESTS)
    add_executable(minixgb_tests tests/test_minixgb.cpp)
    target_link_libraries(minixgb_tests PRIVATE MiniXGBoost::minixgb)
    add_test(NAME minixgb_tests COMMAND minixgb_tests)
endif()

install(
    TARGETS minixgb
    EXPORT MiniXGBoostTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT MiniXGBoostTargets
    FILE MiniXGBoostTargets.cmake
    NAMESPACE MiniXGBoost::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniXGBoost
)

configure_package_config_file(
    cmake/MiniXGBoostConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MiniXGBoostConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniXGBoost
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MiniXGBoostConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/MiniXGBoostConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/MiniXGBoostConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniXGBoost
)
