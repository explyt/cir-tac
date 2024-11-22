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
#include <llvm/Support/ErrorHandling.h>

#include <cinttypes>
#include <iostream>
#include <vector>

using IDToType = llvm::DenseMap<uint64_t, mlir::Type>;
using TypeCache = llvm::DenseMap<mlir::Type, uint64_t>;

using IDToOperation = llvm::DenseMap<uint64_t, mlir::Operation *>;
using OperationCache = llvm::DenseMap<mlir::Operation *, uint64_t>;

uint64_t internType(TypeCache &cache, mlir::Type type) {
  if (!cache.contains(type)) {
    cache[type] = cache.size();
  }
  return cache.at(type);
}

uint64_t internOperation(OperationCache &cache, mlir::Operation *operation) {
  if (!cache.contains(operation)) {
    cache[operation] = cache.size();
  }
  return cache.at(operation);
}

void serializeOperation(mlir::Operation &inst, protocir::CIROp *pInst,
                        protocir::CIRModuleID pModuleID, TypeCache &typeCache,
                        OperationCache &opCache) {
  auto instLine = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
      .Case<cir::AllocaOp>(
          [instLine, pInst, pModuleID, &typeCache](cir::AllocaOp op) {
            protocir::CIRAllocaOp pAllocaOp;
            pAllocaOp.mutable_base()->set_line(instLine);
            protocir::CIRTypeID pTypeID;
            pTypeID.mutable_module_id()->CopyFrom(pModuleID);
            pTypeID.set_id(internType(typeCache, op.getAllocaType()));
            pAllocaOp.mutable_alloca_type()->CopyFrom(pTypeID);
            if (op.getAlignment().has_value()) {
              pAllocaOp.set_alignment(op.getAlignment().value());
            }
            pInst->mutable_alloca()->CopyFrom(pAllocaOp);
          })
      .Case<cir::BinOp>(
          [instLine, pInst, pModuleID, &typeCache](cir::BinOp op) {
            protocir::CIRBinOp pBinOp;
            pBinOp.mutable_base()->set_line(instLine);
          })
      .Case<cir::LoadOp>(
          [instLine, pInst, pModuleID, &typeCache, &opCache](cir::LoadOp op) {
            protocir::CIRLoadOp pLoadOp;
            pLoadOp.mutable_base()->set_line(instLine);
            auto addressLine =
                internOperation(opCache, op.getAddr().getDefiningOp());
            pLoadOp.mutable_address()->set_line(addressLine);
            auto resultTypeID = internType(typeCache, op.getResult().getType());
            pLoadOp.mutable_result_type()->set_id(resultTypeID);
          })
      .Case<cir::StoreOp>(
          [instLine, pInst, pModuleID, &typeCache, &opCache](cir::StoreOp op) {
            protocir::CIRStoreOp pStoreOp;
          })
      .Case<cir::ConstantOp>([instLine, pInst, pModuleID, &typeCache,
                              &opCache](cir::ConstantOp op) {
        protocir::CIRConstantOp pConstantOp;
      })
      .Case<cir::CallOp>([instLine, pInst, pModuleID, &typeCache, &opCache](
                             cir::CallOp op) { protocir::CIRCallOp pCallOp; })
      .Case<cir::ReturnOp>(
          [instLine, pInst, pModuleID, &typeCache, &opCache](cir::ReturnOp op) {
            protocir::CIRReturnOp pReturnOp;
          })
      .Default([](mlir::Operation *) { llvm_unreachable("NIY"); });
}

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<cir::CIRDialect>();

  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();

  std::filesystem::path relPath = argv[1];

  auto absPath = std::filesystem::absolute(relPath);
  if (!std::filesystem::exists(absPath)) {
    throw std::runtime_error("missing path given");
  }

  mlir::ParserConfig parseConfig(&context);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(relPath.c_str(), parseConfig);
  protocir::CIRModule pModule;
  protocir::CIRModuleID pModuleID;
  std::string moduleId = "myModule";
  *pModuleID.mutable_id() = moduleId;
  pModule.mutable_id()->CopyFrom(pModuleID);
  unsigned long func_idx = 0;
  IDToType indexToType;
  TypeCache typeCache;
  auto &bodyRegion = (*module).getBodyRegion();
  for (auto &bodyBlock : bodyRegion) {
    for (auto &func : bodyBlock) {
      protocir::CIRFunction pFunction;
      protocir::CIRFunctionID pFunctionID;
      pFunction.mutable_id()->mutable_module_id()->CopyFrom(pModuleID);
      pFunction.mutable_id()->set_id(++func_idx);
      IDToOperation indexToOp;
      OperationCache opCache;
      for (auto &block : cast<cir::FuncOp>(func).getFunctionBody()) {
        for (auto &inst : block) {
          auto pInst = pFunction.add_operations();
          serializeOperation(inst, pInst, pModuleID, typeCache, opCache);
        }
      }
    }
  }
}
