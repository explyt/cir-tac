#include "CXX/CIRModule.h"
#include "CXX/CIRReader.h"

#include "proto/model.pb.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Passes.h>

#include <mlir/Dialect/DLTI/DLTIDialect.h.inc>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>

#include <iostream>

int main(int argc, char *argv[]) {
  CIRReader reader;
  auto module = reader.loadFromFile(argv[1]);
  protocir::CIRModule pModule;
  protocir::CIRModuleID pModuleID;
  std::string moduleId = "myModule";
  pModuleID.set_allocated_id(&moduleId);
  pModule.set_allocated_id(&pModuleID);
  unsigned long func_idx = 0;
  for (auto func : module.functionsList()) {
    protocir::CIRFunction pFunction;
    protocir::CIRFunctionID pFunctionID;
    pFunctionID.set_id(++func_idx);
    pFunctionID.set_allocated_module_id(&pModuleID);
    pFunction.set_allocated_id(&pFunctionID);
    unsigned long inst_line = 0;
    for (auto inst : func.instructionsList()) {
      auto pInst = pFunction.add_operations();
      protocir::CIRAllocaOp pAllocaOp;
      protocir::CIROpID pOpID;
      pOpID.set_line(inst_line++);
      pAllocaOp.set_allocated_base(&pOpID);
      pInst->set_allocated_alloca(&pAllocaOp);
    }
  }
}
