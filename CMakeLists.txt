cmake_minimum_required(VERSION 3.20)

project(mlit-c-test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED CLANGIR_BUILD_DIR)
  message(FATAL_ERROR "CLANGIR_BUILD_DIR must be specified.")
endif()

# Set up clangir library
find_package(MLIR REQUIRED HINTS ${CLANGIR_BUILD_DIR})
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(CLANG REQUIRED HINTS ${CLANGIR_BUILD_DIR})
message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${CLANG_DIR}")

# Spread includes to entire project
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

include_directories(include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)

# add_executable(mlir-test main.cpp)
# target_link_libraries(mlir-test CIRCAPI)

# CAPI test
# add_executable(mlir-c-test main.c)
# target_link_libraries(mlir-c-test CIRCAPI)

# FIXME: Link libraries to single one with `ar`
install(TARGETS CIRCAPIImpl CIRCAPI PUBLIC_HEADER)
install(DIRECTORY include/CAPI DESTINATION include)
install(FILES include/CIRInstOpCode.h DESTINATION include)
