#pragma once

#include "CAPI/CIRInst.h"
#include "CAPI/CIRInstAPI.h"

#include "CXX/CIRFunction.h"
#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/Casting.h>

struct CIRFunctionRef CIRCallOpCalledFunction(struct CIRInstRef instRef) {
  auto &cirInst = CIRInst::fromRef(instRef);
  auto cirCallInst = cirInst.get<mlir::cir::CallOp>();
  auto called = cirCallInst.resolveCallable();

  return CIRFunction(llvm::dyn_cast<mlir::cir::FuncOp>(called),
                     CIRModule::fromRef(instRef.moduleInnerRef))
      .toRef();
}
