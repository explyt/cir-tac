#pragma once

#include "CAPI/CIRInst.h"
#include "CXX/CIRFunction.h"
#include "CXX/CIRInst.h"
#include <clang/CIR/Dialect/IR/CIRDialect.h>

struct CIRFunctionRef CIRCallOpCalledFunction(struct CIRInstRef instRef) {
  auto &cirInst = CIRInst::fromRef(instRef);
  auto cirCallInst = cirInst.get<mlir::cir::CallOp>();
  auto called = cirCallInst.resolveCallable();

  auto &theModule = reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(
      instRef.innerModuleRef);

  return CIRFunction(*called, theModule).toRef();
}
