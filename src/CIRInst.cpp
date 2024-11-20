#include "CXX/CIRInst.h"
#include "CXX/CIRModule.h"

#include "CIRInstOpCode.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include <llvm/ADT/TypeSwitch.h>

#define CASE(OP_KIND)                                                          \
  Case<cir::OP_KIND>([](cir::OP_KIND) { return CIROpCode::OP_KIND; })

CIROpCode CIRInst::opcode() const {
  CIROpCode result =
      llvm::TypeSwitch<mlir::Operation *, CIROpCode>(&inst)
          .Case<cir ::AllocaOp>(
              [](cir ::AllocaOp) { return CIROpCode ::AllocaOp; })
          .CASE(BinOp)
          .CASE(LoadOp)
          .CASE(StoreOp)
          .CASE(ConstantOp)
          .CASE(CallOp)
          .CASE(ReturnOp)
          .Default([](mlir::Operation *op) { return CIROpCode::UnknownOp; });

  return CIROpCode::UnknownOp;
}
#undef CASE

const CIRInst CIRInst::fromRef(CIRInstRef instRef) {
  auto &operation = *reinterpret_cast<mlir::Operation *>(instRef.innerRef);
  return CIRInst(operation, CIRFunction::fromRef(instRef.functionInnerRef));
}

CIRInstRef CIRInst::toRef() const {
  return CIRInstRef{reinterpret_cast<uintptr_t>(&inst), owner.toRef(),
                    opcode()};
}
