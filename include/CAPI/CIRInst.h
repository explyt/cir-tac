#pragma once

#include <stdlib.h>

#include "CAPI/CIRModule.h"
#include "CIRInstOpCode.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRInstRef {
  uintptr_t innerRef;
  struct CIRModuleRef moduleInnerRef;
  enum CIROpCode opcode;
};

#ifdef __cplusplus
}
#endif
