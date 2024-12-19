set(LIBS
    MLIRCIR
    MLIRParser
    proto
)

add_library(protoSerializer
  EnumsSerializerGen.cpp
  SerializerGen.cpp
)

target_link_libraries(protoSerializer ${LIBS})
