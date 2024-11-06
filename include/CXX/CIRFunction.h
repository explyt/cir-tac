#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"

#include "CXX/CIRInst.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>

#include <cinttypes>
#include <optional>
#include <tuple>

class CIRModule;

class CIRFunction {
public:
  CIRFunction(mlir::Operation &function, const CIRModule &theModule)
      : function(function), theModule(theModule) {
    std::ignore = instructionsList();
  }

  const std::vector<CIRInst> &instructionsList() const;

  static const CIRFunction fromRef(struct CIRFunctionRef ref);
  CIRFunctionRef toRef() const;

  const char *getName() const {
    auto funcOp = llvm::dyn_cast<mlir::cir::FuncOp>(function);
    return funcOp.getName().data();
  }

private:
  mutable std::optional<std::vector<CIRInst>> instructions;
  mlir::Operation &function;
  const CIRModule &theModule;
};
