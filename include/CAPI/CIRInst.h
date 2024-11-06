#pragma once

#include <stdlib.h>

#include "CAPI/CIRFunction.h"
#include "CIRInstOpCode.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRInstRef {
  uintptr_t innerRef;
  struct CIRFunctionRef functionInnerRef;
  enum CIROpCode opcode;
};

#ifdef __cplusplus
}
#endif
