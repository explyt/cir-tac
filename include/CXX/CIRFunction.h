#pragma once

#include "CAPI/CIRInst.h"
#include "CXX/CIRInst.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>

#include <cinttypes>
#include <optional>
#include <tuple>

class CIRFunction {
public:
  CIRFunction(mlir::Operation &function,
              const mlir::OwningOpRef<mlir::ModuleOp> &theModule)
      : function(function), theModule(theModule) {
    std::ignore = instructionsList();
  }

  const std::vector<CIRInst> &instructionsList() const;

  static const CIRFunction fromRef(struct CIRFunctionRef ref) {
    auto &func = *reinterpret_cast<mlir::Operation *>(ref.innerRef);
    auto &theModule = *reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> *>(
        ref.innerModuleRef);
    return CIRFunction(func, theModule);
  }

  CIRFunctionRef toRef() const {
    return CIRFunctionRef{reinterpret_cast<uintptr_t>(&function),
                          reinterpret_cast<uintptr_t>(&theModule),
                          instructions->size()};
  }

  const char *getName() const {
    auto funcOp = llvm::dyn_cast<mlir::cir::FuncOp>(function);
    return funcOp.getName().data();
  }

private:
  mutable std::optional<std::vector<CIRInst>> instructions;
  mlir::Operation &function;
  const mlir::OwningOpRef<mlir::ModuleOp> &theModule;
};
