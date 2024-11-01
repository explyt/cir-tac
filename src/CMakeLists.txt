add_library(CIRCAPI 
  CIRReader.cpp
  CIRModule.cpp
  CIRFunction.cpp
  CAPI/CIR.cpp
)

target_link_libraries(CIRCAPI MLIRCIR MLIRParser MLIRSupport)
