#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/DLTI/DLTIDialect.h.inc>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <clang/CIR/Passes.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <iostream>

#include "CXX/CIRModule.h"
#include "CXX/CIRReader.h"

int main(int argc, char *argv[]) {
  // mlir::DialectRegistry registry;
  // registry.insert<cir::CIRDialect, mlir::DLTIDialect,
  //                 mlir::LLVM::LLVMDialect>();

  // mlir::MLIRContext ctx(registry, mlir::MLIRContext::Threading::DISABLED);

  // mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/true);

  // auto module = parseSourceFile<mlir::Operation *>(argv[1], parseConfig);

  // std::cout << module->getNumRegions() << std::endl;

  // auto &region = module->getRegion(0);

  // int cnt = 0;
  // for (auto &block : region) {
  //   auto &funcList = block.getOperations();
  //   for (auto &func : funcList) {
  //     auto &funcRegion = func.getRegion(0);
  //     auto &funcBody = funcRegion.getBlocks().front().getOperations();
  //     for (auto &op : funcBody) {
  //       // if (dynamic_cast<cir::AllocaOp *>(op)) {
  //       // }
  //       if (llvm::isa<cir::AllocaOp>(op)) {
  //         llvm::errs() << "ALLOCA!\n";
  //       }

  //       // op.getOperand(0)
  //       ++cnt;
  //     }
  //   }
  // }
  CIRReader reader;
  auto module = reader.loadFromFile(argv[1]);
  for (auto func : module.functionsList()) {
    func.instructionsList();
  }

  // std::cout << cnt << std::endl;
  // module->dump();
}