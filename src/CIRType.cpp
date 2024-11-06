#include "CXX/CIRType.h"
#include "CXX/CIRModule.h"

const CIRType CIRType::fromRef(struct CIRTypeRef ref) {
  auto type =
      mlir::Type::getFromOpaquePointer(reinterpret_cast<void *>(ref.innerRef));
  return CIRType(type, CIRModule::fromRef(ref.moduleInnerRef));
}

CIRTypeRef CIRType::toRef() const {
  return CIRTypeRef{reinterpret_cast<uintptr_t>(type.getAsOpaquePointer()),
                    theModule.toRef()};
}

const char *CIRType::getName() const {
  return type.getAbstractType().getName().data();
}

size_t CIRType::size() const { return theModule.dl.getTypeSize(type); }
