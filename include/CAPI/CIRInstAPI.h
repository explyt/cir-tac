#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRType.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

//
struct CIRFunctionRef CIRInstGetParentFunction(struct CIRInstRef instRef);

// AllocaOp
struct CIRTypeRef CIRAllocaOpType(struct CIRInstRef instRef);
size_t CIRAllocaOpAlignment(struct CIRInstRef instRef);
struct CIRInstRef CIRAllocaSize(struct CIRInstRef instRef);

// CallOp
struct CIRFunctionRef CIRCallOpCalledFunction(struct CIRInstRef instRef);

// LoadOp
struct CIRTypeRef CIRLoadOpType(struct CIRInstRef instRef);
struct CIRInstRef CIRLoadOpAddress(struct CIRInstRef instRef);

// StoreOp
struct CIRInstRef CIRStoreOpAddress(struct CIRInstRef instRef);
struct CIRTypeRef CIRStoreOpType(struct CIRInstRef instRef);

struct CIRInstRef CIRStoreOpValue(struct CIRInstRef instRef);

// ReturnOp
struct CIRInstRef CIRReturnOpGetValue(struct CIRInstRef instRef);
struct CIRTypeRef CIRReturnOpGetType(struct CIRInstRef instRef);

#ifdef __cplusplus
}
#endif
