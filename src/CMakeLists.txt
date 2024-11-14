add_library(CIRCAPI STATIC
  CIRFunction.cpp
  CIRInst.cpp
  CIRModule.cpp
  CIRReader.cpp
  CIRType.cpp
)

add_subdirectory(CAPI)

# TODO: Figure out which libraries are required
target_link_libraries(CIRCAPIImpl CIRCAPI MLIRCIR MLIRParser MLIRSupport)
