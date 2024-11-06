#include "CXX/CIRFunction.h"
#include <CXX/CIRModule.h>

#include <cassert>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <type_traits>

CIRModule::CIRModule(mlir::OwningOpRef<mlir::ModuleOp> &&refModule)
    : dl(*refModule), theModule(std::move(refModule)) {
  std::ignore = functionsList();
}

const std::vector<CIRFunction> &CIRModule::functionsList() const {
  if (!functions.has_value()) {
    functions.emplace();
    auto &bodyRegion = (*theModule).getBodyRegion();
    for (auto &bodyBlock : bodyRegion) {
      for (auto &function : bodyBlock) {
        assert(llvm::isa<mlir::cir::FuncOp>(&function));
        functions->emplace_back(function, *this);
      }
    }
  }
  return *functions;
}
