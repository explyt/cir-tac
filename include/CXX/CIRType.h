#pragma once

#include "CAPI/CIRModule.h"
#include "CAPI/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRAttrs.h>
#include <clang/CIR/Dialect/IR/CIROpsEnums.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>

#include <cinttypes>

class CIRModule;

class CIRType {
public:
  CIRType(mlir::Type type, const CIRModule &theModule)
      : type(type), theModule(theModule) {}

  static const CIRType fromRef(struct CIRTypeRef ref);
  CIRTypeRef toRef() const;

  const char *getName() const;
  size_t size() const;

private:
  mlir::Type type;
  const CIRModule &theModule;
};
