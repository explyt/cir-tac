#include "CXX/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CIRInstOpCode.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/Support/TypeID.h>

const std::vector<CIRInst> &CIRFunction::instructionsList() const {
  if (!instructions.has_value()) {
    instructions.emplace();
    auto cirFunction = dyn_cast<mlir::cir::FuncOp>(function);
    for (auto &block : cirFunction.getFunctionBody()) {
      for (auto &op : block) {
        instructions->emplace_back(op, theModule);
      }
    }
  }

  return *instructions;
}