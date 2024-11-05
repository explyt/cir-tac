#pragma once

#include <inttypes.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRTypeRef {
  uintptr_t innerRef;
  uintptr_t innerModuleRef;
};

const char *CIRTypeGetName(struct CIRTypeRef ref);

size_t CIRGetTypeSize(struct CIRTypeRef ref);

#ifdef __cplusplus
}
#endif
