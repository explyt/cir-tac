#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRModule.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CIRFunctionRef CIRModuleGetFunction(struct CIRModuleRef moduleRef,
                                           size_t idx);

#ifdef __cplusplus
}
#endif
