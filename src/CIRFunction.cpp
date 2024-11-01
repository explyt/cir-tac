#include "CXX/CIRFunction.h"
#include "CAPI/CIRInst.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/Support/TypeID.h>


CIRInstOpcode determineOpCode(mlir::Operation &op) {
  auto opTypeID = op.getName().getTypeID();

  using mlir::TypeID;
  using namespace mlir::cir;

  if (opTypeID == TypeID::get<AllocaOp>()) {
    llvm::errs() << "alloca";
  }
  if (opTypeID == TypeID::get<LoadOp>()) {
    llvm::errs() << "load";
  }
  if (opTypeID == TypeID::get<StoreOp>()) {
    llvm::errs() << "store";
  }
  if (opTypeID == TypeID::get<ConstantOp>()) {
    llvm::errs() << "constant";
  }
  if (opTypeID == TypeID::get<ReturnOp>()) {
    llvm::errs() << "return";
  }
  if (opTypeID == TypeID::get<BinOp>()) {
    llvm::errs() << "binop";
  }
  if (opTypeID == TypeID::get<CallOp>()) {
    llvm::errs() << "call";
  }

  llvm::errs() << "\n";
  return {};
}

std::vector<CIRInst> CIRFunction::instructionsList() const {
  std::vector<CIRInst> result;

  for (auto &region : function->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        determineOpCode(op);
      }
    }
  }

  return result;
}