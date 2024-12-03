set(LIBS
    protoSerializer
)

add_executable(cir-ser-proto cir-ser-proto.cpp)

target_link_libraries(cir-ser-proto ${LIBS})
