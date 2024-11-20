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

#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

using indexed_types = std::unordered_map<uint64_t, mlir::Type>;
using intern_type_cache = std::unordered_map<mlir::Type, uint64_t>;

uint64_t internType(intern_type_cache &cache, mlir::Type type) {
  if (!cache.contains(type)) {
    cache[type] = cache.size();
  }
  return cache.at(type);
}

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<cir::CIRDialect>();

  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();

  std::filesystem::path apath = argv[1];

  auto absPath = std::filesystem::absolute(apath);
  if (!std::filesystem::exists(absPath)) {
    throw std::runtime_error("missing path given");
  }

  mlir::ParserConfig parseConfig(&context);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(apath.c_str(), parseConfig);
  protocir::CIRModule pModule;
  protocir::CIRModuleID pModuleID;
  std::string moduleId = "myModule";
  pModuleID.set_allocated_id(&moduleId);
  pModule.set_allocated_id(&pModuleID);
  unsigned long func_idx = 0;
  std::unordered_map<uint64_t, mlir::Type> indexToType;
  std::unordered_map<mlir::Type, uint64_t> internCache;
  auto &bodyRegion = (*module).getBodyRegion();
  for (auto &bodyBlock : bodyRegion) {
    for (auto &func : bodyBlock) {
      protocir::CIRFunction pFunction;
      protocir::CIRFunctionID pFunctionID;
      pFunctionID.set_id(++func_idx);
      pFunctionID.set_allocated_module_id(&pModuleID);
      pFunction.set_allocated_id(&pFunctionID);
      unsigned long inst_line = 0;
      for (auto &block : cast<cir::FuncOp>(func).getFunctionBody()) {
        for (auto &inst : block) {
          auto pInst = pFunction.add_operations();
          llvm::TypeSwitch<mlir::Operation *>(&inst)
              .Case<cir::AllocaOp>([](cir::AllocaOp) {})
              .Default([](mlir::Operation *) {});
          protocir::CIRAllocaOp pAllocaOp;
          protocir::CIROpID pOpID;
          pOpID.set_line(inst_line++);
          pAllocaOp.set_allocated_base(&pOpID);
          pInst->set_allocated_alloca(&pAllocaOp);
        }
      }
    }
  }
}
