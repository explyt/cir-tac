add_library(CIRCAPI 
  CIRFunction.cpp
  CIRInst.cpp
  CIRModule.cpp
  CIRReader.cpp
  CIRType.cpp
)

add_subdirectory(CAPI)

# TODO: Figure out which libraries are required
target_link_libraries(CIRCAPI CIRCAPIImpl MLIRCIR MLIRParser MLIRSupport)

set(CAPI_HEADERS
  include/CAPI/CIR.h
  include/CAPI/CIRFunction.h
  include/CAPI/CIRFunctionAPI.h
  include/CAPI/CIRInst.h
  include/CAPI/CIRInstAPI.h
  include/CAPI/CIRModule.h
  include/CAPI/CIRModuleAPI.h
  include/CAPI/CIRType.h
  include/CAPI/CIRTypeAPI.h
)
set_target_properties(CIRCAPI PROPERTIES PUBLIC_HEADER "${CAPI_HEADERS}")