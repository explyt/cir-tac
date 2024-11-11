#include "CXX/CIRFunction.h"
#include "CXX/CIRModule.h"

#include "CAPI/CIRInst.h"
#include "CIRInstOpCode.h"
#include "CXX/CIRType.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/Support/TypeID.h>

const std::vector<CIRInst> &CIRFunction::instructionsList() const {
  if (!instructions.has_value()) {
    instructions.emplace();
    for (auto &block : function.getFunctionBody()) {
      for (auto &op : block) {
        instructions->emplace_back(op, *this);
      }
    }
  }

  return *instructions;
}

const CIRFunction CIRFunction::fromRef(struct CIRFunctionRef ref) {
  auto func = mlir::cir::FuncOp::getFromOpaquePointer(
      reinterpret_cast<void *>(ref.innerRef));
  return CIRFunction(func, CIRModule::fromRef(ref.moduleInnerRef));
}

CIRFunctionRef CIRFunction::toRef() const {
  return CIRFunctionRef{
      reinterpret_cast<uintptr_t>(function.getAsOpaquePointer()),
      theModule.toRef(), instructions->size(), function.getNumArguments()};
}

CIRType CIRFunction::getReturnType() const {
  return CIRType(function.getFunctionType().getReturnType(), theModule);
}

CIRType CIRFunction::getArgumentType(std::size_t argNum) const {
  return CIRType(function.getArgumentTypes()[argNum], theModule);
}
