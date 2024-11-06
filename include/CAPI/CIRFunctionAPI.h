#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRInstRef CIRFunctionGetInst(struct CIRFunctionRef funcRef, size_t idx);

const char *CIRFunctionGetName(struct CIRFunctionRef funcRef);

#ifdef __cplusplus
}
#endif
