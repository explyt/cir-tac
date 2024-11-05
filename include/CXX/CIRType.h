#pragma once

#include "CAPI/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRAttrs.h>
#include <clang/CIR/Dialect/IR/CIROpsEnums.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <llvm/Support/raw_ostream.h>

#include <cinttypes>

class CIRType {
public:
  CIRType(mlir::Type type, mlir::OwningOpRef<mlir::ModuleOp> &theModule)
      : type(type), theModule(theModule) {}

  static const CIRType fromRef(struct CIRTypeRef ref) {
    auto type = mlir::Type::getFromOpaquePointer(
        reinterpret_cast<void *>(ref.innerRef));
    auto &theModule = *reinterpret_cast<mlir::OwningOpRef<mlir::ModuleOp> *>(
        ref.innerModuleRef);
    return CIRType(type, theModule);
  }

  CIRTypeRef toRef() const {
    return CIRTypeRef{reinterpret_cast<uintptr_t>(type.getAsOpaquePointer())};
  }

  const char *getName() const {
    return type.getAbstractType().getName().data();
  }

  size_t size() const { return 0; }

private:
  mlir::Type type;
  const mlir::OwningOpRef<mlir::ModuleOp> &theModule;
};
