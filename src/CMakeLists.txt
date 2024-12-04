set(LIBS
    MLIRCIR
    MLIRParser
    proto
)

add_library(protoSerializer
  SerializerGen.cpp
)

target_link_libraries(protoSerializer ${LIBS})
