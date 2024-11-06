#include "CAPI/CIRInst.h"
#include "CAPI/CIRInstAPI.h"
#include "CAPI/CIRType.h"

#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"
#include "CXX/CIRType.h"

#include "CIRInstOpCode.h"

#include <cassert>
#include <clang/CIR/Dialect/IR/CIRDialect.h>

CIRTypeRef CIRAllocaOpType(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  auto type = cirAllocaOp.getAllocaType();

  return CIRType(type, CIRModule::fromRef(instRef.moduleInnerRef)).toRef();
}

CIRInstRef CIRAllocaSize(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  // FIXME:
  assert(false && "NYI");

  if (cirAllocaOp.isDynamic()) {
    auto opSize = cirAllocaOp.getDynAllocSize();

    return CIRInst(*opSize.getDefiningOp(),
                   CIRModule::fromRef(instRef.moduleInnerRef))
        .toRef();
  }
}

size_t CIRAllocaOpAlignment(struct CIRInstRef instRef) {
  auto cirOp = CIRInst::fromRef(instRef);
  auto cirAllocaOp = cirOp.get<mlir::cir::AllocaOp>();
  return cirAllocaOp.getAlignment().value_or(0);
}
