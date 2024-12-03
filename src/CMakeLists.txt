set(LIBS
    MLIRCIR
    MLIRParser
    proto
)

add_library(protoSerializer
  Serializer.cpp
)

target_link_libraries(protoSerializer ${LIBS})
