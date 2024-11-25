set(TOOL_FILES
    cir-ser-proto.cpp
)

add_executable(cir-ser-proto ${TOOL_FILES})

target_link_libraries(cir-ser-proto PRIVATE
    MLIRCIR
    MLIRParser
    proto
)

# Include necessary directories
target_include_directories(cir-ser-proto PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
