#pragma once

#include "CAPI/CIRModule.h"
#include "CAPI/CIRType.h"
#include "CXX/CIRModule.h"

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
  CIRType(mlir::Type type, const CIRModule &theModule)
      : type(type), theModule(theModule) {}

  static const CIRType fromRef(struct CIRTypeRef ref) {
    auto type = mlir::Type::getFromOpaquePointer(
        reinterpret_cast<void *>(ref.innerRef));
    return CIRType(type, CIRModule::fromRef(ref.moduleInnerRef));
  }

  CIRTypeRef toRef() const {
    return CIRTypeRef{reinterpret_cast<uintptr_t>(type.getAsOpaquePointer()),
                      theModule.toRef()};
  }

  const char *getName() const {
    return type.getAbstractType().getName().data();
  }

  size_t size() const { return theModule.dl.getTypeSize(type); }

private:
  mlir::Type type;
  const CIRModule &theModule;
};
