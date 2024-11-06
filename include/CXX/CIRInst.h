#pragma once

#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"
#include "CIRInstOpCode.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <cinttypes>
#include <mlir/IR/OwningOpRef.h>

class CIRModule;

class CIRInst {
public:
  CIRInst(mlir::Operation &inst, const CIRModule &moduleRef)
      : inst(inst), theModule(moduleRef) {}

  template <typename T> const T get() const {
    return llvm::dyn_cast<const T>(&inst);
  }

  CIROpCode opcode() const;

  static const CIRInst fromRef(CIRInstRef instRef);
  CIRInstRef toRef() const;

private:
  mlir::Operation &inst;
  const CIRModule &theModule;
};
