#pragma once

#include "CXX/CIRInst.h"
#include <clang/CIR/Dialect/IR/CIRDialect.h>
class CIRFunction {
public:
  CIRFunction(const mlir::cir::FuncOp &function) : function(function) {}

  std::vector<CIRInst> instructionsList() const;
private:
  mlir::cir::FuncOp function;
};
