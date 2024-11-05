#pragma once

#include <inttypes.h>
#include <stdlib.h>

#include "CAPI/CIRInst.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRModuleRef {
  uintptr_t innerRef;
  size_t functionsNum;
};

struct CIRFunctionRef CIRModuleGetFunction(struct CIRModuleRef moduleRef,
                                           size_t idx);

#ifdef __cplusplus
}
#endif
