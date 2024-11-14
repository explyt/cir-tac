#include "CAPI/CIRInst.h"
#include "CAPI/CIRInstAPI.h"

#include "CAPI/CIRType.h"
#include "CIRInstOpCode.h"

#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"
#include "CXX/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>

struct CIRInstRef CIRReturnOpGetValue(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirReturnOp = cirInst.get<cir::ReturnOp>();
  if (cirReturnOp->getNumOperands() == 0) {
    return CIRInstRef{0, instRef.functionInnerRef, UnknownOp};
  }

  assert(cirReturnOp->getNumOperands() == 1);
  auto returnOp = cirReturnOp->getOperand(0).getDefiningOp();
  return CIRInst(*returnOp, CIRFunction::fromRef(instRef.functionInnerRef))
      .toRef();
}

struct CIRTypeRef CIRReturnOpGetType(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirReturnOp = cirInst.get<cir::ReturnOp>();

  auto owner = CIRFunction::fromRef(instRef.functionInnerRef);
  return owner.getReturnType().toRef();
}
