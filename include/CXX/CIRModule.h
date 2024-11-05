#pragma once

#include "CAPI/CIRModule.h"
#include "CXX/CIRFunction.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <filesystem>
#include <optional>

class CIRModule {
public:
  CIRModule(mlir::OwningOpRef<mlir::ModuleOp> &&theModule);

  const std::vector<CIRFunction> &functionsList() const;

  static const CIRModule &fromRef(CIRModuleRef moduleRef) {
    return *reinterpret_cast<CIRModule *>(moduleRef.innerRef);
  }

  CIRModuleRef toRef() const {
    return CIRModuleRef{reinterpret_cast<uintptr_t>(this), functions->size()};
  }

private:
  mutable std::optional<std::vector<CIRFunction>> functions;
  mlir::OwningOpRef<mlir::ModuleOp> theModule;
  mlir::DataLayout dl;
};
