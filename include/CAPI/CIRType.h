#pragma once

#include "CAPI/CIRModule.h"

#include <inttypes.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRTypeRef {
  uintptr_t innerRef;
  struct CIRModuleRef moduleInnerRef;
};

#ifdef __cplusplus
}
#endif
