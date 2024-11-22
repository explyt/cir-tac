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
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>

#include <cinttypes>
#include <iostream>
#include <vector>

using indexed_types = llvm::DenseMap<uint64_t, mlir::Type>;
using intern_type_cache = llvm::DenseMap<mlir::Type, uint64_t>;

uint64_t internType(intern_type_cache &cache, mlir::Type type) {
  if (!cache.contains(type)) {
    cache[type] = cache.size();
  }
  return cache.at(type);
}

void serializeOperation(mlir::Operation &inst, protocir::CIROp *pInst,
                        unsigned long &inst_line,
                        protocir::CIRModuleID pModuleID,
                        intern_type_cache &internCache) {
  llvm::TypeSwitch<mlir::Operation *>(&inst)
      .Case<cir::AllocaOp>(
          [&inst_line, pInst, pModuleID, &internCache](cir::AllocaOp op) {
            protocir::CIRAllocaOp pAllocaOp;
            protocir::CIROpID pOpID;
            pOpID.set_line(inst_line++);
            pAllocaOp.mutable_base()->CopyFrom(pOpID);
            protocir::CIRTypeID pTypeID;
            pTypeID.mutable_module_id()->CopyFrom(pModuleID);
            pTypeID.set_id(internType(internCache, op.getAllocaType()));
            pAllocaOp.mutable_type()->CopyFrom(pTypeID);
            if (op.getAlignment().has_value()) {
              pAllocaOp.set_alignment(op.getAlignment().value());
            }
            pInst->mutable_alloca()->CopyFrom(pAllocaOp);
          })
      .Default([](mlir::Operation *) {});
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
  *pModuleID.mutable_id() = moduleId;
  pModule.mutable_id()->CopyFrom(pModuleID);
  unsigned long func_idx = 0;
  indexed_types indexToType;
  intern_type_cache internCache;
  auto &bodyRegion = (*module).getBodyRegion();
  for (auto &bodyBlock : bodyRegion) {
    for (auto &func : bodyBlock) {
      protocir::CIRFunction pFunction;
      protocir::CIRFunctionID pFunctionID;
      pFunction.mutable_id()->mutable_module_id()->CopyFrom(pModuleID);
      pFunction.mutable_id()->set_id(++func_idx);
      unsigned long inst_line = 0;
      for (auto &block : cast<cir::FuncOp>(func).getFunctionBody()) {
        for (auto &inst : block) {
          auto pInst = pFunction.add_operations();
          serializeOperation(inst, pInst, inst_line, pModuleID, internCache);
        }
      }
    }
  }
}
