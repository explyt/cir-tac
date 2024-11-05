#pragma once

#include <stdlib.h>

#include "CAPI/CIRInst.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRFunctionRef {
  uintptr_t innerRef;
  uintptr_t innerModuleRef;
  size_t instructionsNum;
};

struct CIRInstRef CIRFunctionGetInst(struct CIRFunctionRef funcRef, size_t idx);

const char *CIRFunctionGetName(struct CIRFunctionRef funcRef);

#ifdef __cplusplus
}
#endif
