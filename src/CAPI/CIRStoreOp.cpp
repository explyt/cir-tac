#pragma once

#include "CAPI/CIRInst.h"
#include "CAPI/CIRInstAPI.h"

#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>

struct CIRInstRef CIRStoreOpAddress(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirStoreOp = cirInst.get<mlir::cir::StoreOp>();

  auto address = cirStoreOp.getAddr();
  auto opAddress = address.getDefiningOp();

  return CIRInst(*opAddress, CIRModule::fromRef(instRef.moduleInnerRef))
      .toRef();
}

struct CIRInstRef CIRStoreOpValue(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirStoreOp = cirInst.get<mlir::cir::StoreOp>();

  auto value = cirStoreOp.getValue();
  auto opValue = value.getDefiningOp();

  return CIRInst(*opValue, CIRModule::fromRef(instRef.moduleInnerRef)).toRef();
}
