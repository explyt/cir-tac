#pragma once

#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"

#include "CXX/CIRInst.h"
#include "CXX/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Operation.h>

#include <cinttypes>
#include <optional>
#include <tuple>

class CIRModule;

class CIRFunction {
public:
  CIRFunction(mlir::cir::FuncOp function, const CIRModule &theModule)
      : function(function), theModule(theModule) {
    std::ignore = instructionsList();
  }

  const std::vector<CIRInst> &instructionsList() const;

  static const CIRFunction fromRef(struct CIRFunctionRef ref);
  CIRFunctionRef toRef() const;

  const char *getName() const { return function.getName().data(); }

  CIRType getReturnType() const;

private:
  // FIXME: use cache in module or like that
  mutable std::optional<std::vector<CIRInst>> instructions;

  // TODO: FuncOp::getName() is non const-qualified o_0
  mutable mlir::cir::FuncOp function;
  const CIRModule &theModule;
};
