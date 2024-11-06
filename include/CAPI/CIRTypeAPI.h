#pragma once

#include <CAPI/CIRType.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *CIRTypeGetName(struct CIRTypeRef ref);

size_t CIRTypeGetSize(struct CIRTypeRef ref);

#ifdef __cplusplus
}
#endif
