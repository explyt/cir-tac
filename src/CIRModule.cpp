#include "CXX/CIRFunction.h"
#include <CXX/CIRModule.h>

#include <cassert>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <type_traits>

CIRModule::CIRModule(mlir::OwningOpRef<mlir::ModuleOp> &&recModule)
    : theModule(std::move(recModule)) {}

std::vector<CIRFunction> CIRModule::functionsList() const {
  std::vector<CIRFunction> result;

  auto &bodyRegion = (*theModule).getBodyRegion();
  for (auto &bodyBlock : bodyRegion) {
    for (auto &function : bodyBlock) {
      assert(llvm::isa<mlir::cir::FuncOp>(&function));
      auto rawCIRFunction = llvm::dyn_cast<mlir::cir::FuncOp>(&function);
      result.emplace_back(rawCIRFunction);
    }
  }

  return result;
}