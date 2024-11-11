#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRType.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRInstRef CIRFunctionGetInst(struct CIRFunctionRef funcRef, size_t idx);

const char *CIRFunctionGetName(struct CIRFunctionRef funcRef);

struct CIRTypeRef CIRFunctionGetReturnType(struct CIRFunctionRef funcRef);

struct CIRTypeRef CIRFunctionGetArgumentType(struct CIRFunctionRef funcRef,
                                             size_t argNum);

#ifdef __cplusplus
}
#endif
