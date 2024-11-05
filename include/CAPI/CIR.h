#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include "CAPI/CIRModule.h"
#include "CIRInstOpCode.h"

#ifdef __cplusplus
extern "C" {
#endif

struct CIRReaderRef {
  uintptr_t innerRef;
};

struct CIRReaderRef CIRCreateReader();
void CIRDestroyReader(struct CIRReaderRef ref);

struct CIRModuleRef loadModuleFromFile(struct CIRReaderRef ref,
                                       const char *name);
void CIRDestroyModule(struct CIRModuleRef ref);

#ifdef __cplusplus
}
#endif
