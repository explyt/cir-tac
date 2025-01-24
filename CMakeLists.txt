cmake_minimum_required(VERSION 3.20)

project(cir-tac)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED CLANGIR_BUILD_DIR)
  message(FATAL_ERROR "CLANGIR_BUILD_DIR must be specified.")
endif()

find_package(MLIR REQUIRED HINTS ${CLANGIR_BUILD_DIR})
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(Clang REQUIRED HINTS ${CLANGIR_BUILD_DIR})
message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${CLANG_DIR}")

# Spread includes to entire project
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Found Protobuf ${PROTOBUF_PACKAGE_VERSION}")
message(STATUS "Using Protobuf.cmake in: ${PROTOBUF_DIR}")

add_subdirectory(proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROTOBUF_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(src)
add_subdirectory(tools)
