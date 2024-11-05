#include "CAPI/CIRInst.h"
#include "CAPI/CIRType.h"
#include "CIRInstOpCode.h"
#include "CXX/CIRInst.h"
#include "CXX/CIRType.h"

#include <cassert>
#include <clang/CIR/Dialect/IR/CIRDialect.h>

CIRTypeRef CIRAllocaOpType(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  auto type = cirAllocaOp.getAllocaType();

  auto &theModule = reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(
      instRef.innerModuleRef);

  return CIRType(type, theModule).toRef();
}

CIRInstRef CIRAllocaSize(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  // FIXME:
  assert(false && "NYI");
  auto &theModule = reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(
      instRef.innerModuleRef);

  if (cirAllocaOp.isDynamic()) {
    auto opSize = cirAllocaOp.getDynAllocSize();

    return CIRInst(*opSize.getDefiningOp(), theModule).toRef();
  }
}

size_t CIRAllocaOpAlignment(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  return cirAllocaOp.getAlignment().value_or(0);
}
