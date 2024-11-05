#pragma once

#include "CAPI/CIRInst.h"
#include "CIRInstOpCode.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <cinttypes>
#include <mlir/IR/OwningOpRef.h>
class CIRInst {
public:
  CIRInst(mlir::Operation &inst,
          const mlir::OwningOpRef<mlir::ModuleOp> &theModule)
      : inst(inst), theModule(theModule) {}

  template <typename T> const T get() const {
    return llvm::dyn_cast<const T>(&inst);
  }

  CIROpCode opcode() const;

  static const CIRInst fromRef(CIRInstRef instRef) {
    auto &operation = *reinterpret_cast<mlir::Operation *>(instRef.innerRef);
    auto &theModule =
        *reinterpret_cast<const mlir::OwningOpRef<mlir::ModuleOp> *>(
            instRef.innerModuleRef);
    return CIRInst(operation, theModule);
  }

  CIRInstRef toRef() const {
    return CIRInstRef{reinterpret_cast<uintptr_t>(&inst),
                      reinterpret_cast<uintptr_t>(&theModule), opcode()};
  }

private:
  mlir::Operation &inst;
  const mlir::OwningOpRef<mlir::ModuleOp> &theModule;
};
