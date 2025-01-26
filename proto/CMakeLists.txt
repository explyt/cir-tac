find_package(Protobuf CONFIG REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})

set(PROTOSRC
    enumgen.proto
    model.proto
    opgen.proto
    setup.proto
    type.proto
)

add_library(proto OBJECT ${PROTOSRC})

protobuf_generate(
    TARGET proto
    IMPORT_DIRS .
    PROTOS
    ${PROTOSRC}
)

target_link_libraries(
    proto PRIVATE
    protobuf::libprotobuf
)
