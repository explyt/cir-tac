add_library(CIRCAPI 
  CIRReader.cpp
  CIRModule.cpp
  CIRFunction.cpp
  CIRInst.cpp
)

add_subdirectory(CAPI)

target_link_libraries(CIRCAPI CAPIImpl MLIRCIR MLIRParser MLIRSupport)
