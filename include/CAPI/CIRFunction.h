#pragma once

#include <stdlib.h>

#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRFunctionRef {
  uintptr_t innerRef;
  struct CIRModuleRef moduleInnerRef;
  size_t instructionsNum;
};

#ifdef __cplusplus
}
#endif
