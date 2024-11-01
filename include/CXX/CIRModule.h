#pragma once

#include "CXX/CIRFunction.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <filesystem>

class CIRModule {
public:
  CIRModule(mlir::OwningOpRef<mlir::ModuleOp> &&theModule);

  std::vector<CIRFunction> functionsList() const;
private:
  mlir::OwningOpRef<mlir::ModuleOp> theModule;
};
