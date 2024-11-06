#pragma once

#include <inttypes.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRModuleRef {
  uintptr_t innerRef;
  size_t functionsNum;
};

#ifdef __cplusplus
}
#endif
