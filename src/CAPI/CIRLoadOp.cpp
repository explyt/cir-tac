#include "CAPI/CIRInst.h"
#include "CAPI/CIRInstAPI.h"
#include "CAPI/CIRType.h"

#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"
#include "CXX/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

struct CIRInstRef CIRLoadOpAddress(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirLoadInst = cirInst.get<cir::LoadOp>();

  auto address = cirLoadInst.getAddr();
  auto addressValue = address.getDefiningOp();
  //
  return CIRInst(*addressValue, CIRFunction::fromRef(instRef.functionInnerRef))
      .toRef();
}

struct CIRTypeRef CIRLoadOpType(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirLoadInst = cirInst.get<cir::LoadOp>();

  auto address = cirLoadInst.getAddr();
  auto addressType = address.getType();

  return CIRType(addressType,
                 CIRModule::fromRef(instRef.functionInnerRef.moduleInnerRef))
      .toRef();
}
