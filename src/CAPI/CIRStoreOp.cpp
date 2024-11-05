#pragma once

#include "CAPI/CIRInst.h"
#include "CXX/CIRInst.h"
#include <clang/CIR/Dialect/IR/CIRDialect.h>

struct CIRInstRef CIRStoreOpAddress(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirStoreOp = cirInst.get<mlir::cir::StoreOp>();

  auto address = cirStoreOp.getAddr();
  auto opAddress = address.getDefiningOp();

  auto &theModule = reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(
      instRef.innerModuleRef);

  return CIRInst(*opAddress, theModule).toRef();
}

struct CIRInstRef CIRStoreOpValue(struct CIRInstRef instRef) {
  auto cirInst = CIRInst::fromRef(instRef);
  auto cirStoreOp = cirInst.get<mlir::cir::StoreOp>();

  auto value = cirStoreOp.getValue();
  auto opValue = value.getDefiningOp();

  auto &theModule = reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> &>(
      instRef.innerModuleRef);

  return CIRInst(*opValue, theModule).toRef();
}
