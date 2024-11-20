include(FindProtobuf)
find_package(Protobuf REQUIRED CONFIG)
include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER model.proto)
add_library(proto ${PROTO_HEADER} ${PROTO_SRC})
target_link_libraries(
    proto ${Protobuf_LIBRARIES}
    absl::log_internal_message
    absl::log_internal_check_op
)
