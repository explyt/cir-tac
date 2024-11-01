#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include "CIRInstOpCode.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CIR, cir);

MLIR_CAPI_EXPORTED MlirCIROpCode mlirCIRdetermineOpCode(MlirOperation op);

#ifdef __cplusplus
}
#endif
