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
#include <fstream>
#include <iostream>
#include <vector>

using TypeCache = llvm::DenseMap<mlir::Type, uint64_t>;

using OperationCache = llvm::DenseMap<mlir::Operation *, uint64_t>;

using FunctionCache = llvm::DenseMap<cir::FuncOp, uint64_t>;

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

uint64_t internFunction(FunctionCache &cache, cir::FuncOp operation) {
  if (!cache.contains(operation)) {
    cache[operation] = cache.size();
  }
  return cache.at(operation);
}

void serializeOperation(mlir::Operation &inst, protocir::CIROp *pInst,
                        protocir::CIRModuleID pModuleID, TypeCache &typeCache,
                        OperationCache &opCache, FunctionCache &functionCache

) {
  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
      .Case<cir::AllocaOp>(
          [instID, pInst, pModuleID, &typeCache](cir::AllocaOp op) {
            protocir::CIRAllocaOp pAllocaOp;
            pAllocaOp.mutable_base()->set_id(instID);
            protocir::CIRTypeID pTypeID;
            pTypeID.mutable_module_id()->CopyFrom(pModuleID);
            pTypeID.set_id(internType(typeCache, op.getAllocaType()));
            pAllocaOp.mutable_alloca_type()->CopyFrom(pTypeID);
            if (op.getAlignment().has_value()) {
              pAllocaOp.set_alignment(op.getAlignment().value());
            }
            pInst->mutable_alloca()->CopyFrom(pAllocaOp);
          })
      .Case<cir::BinOp>([instID, pInst, pModuleID, &typeCache](cir::BinOp op) {
        protocir::CIRBinOp pBinOp;
        pBinOp.mutable_base()->set_id(instID);
        pInst->mutable_bin()->CopyFrom(pBinOp);
      })
      .Case<cir::LoadOp>([instID, pInst, pModuleID, &typeCache,
                          &opCache](cir::LoadOp op) {
        protocir::CIRLoadOp pLoadOp;
        pLoadOp.mutable_base()->set_id(instID);
        auto addressID = internOperation(opCache, op.getAddr().getDefiningOp());
        pLoadOp.mutable_address()->set_id(addressID);
        auto resultTypeID = internType(typeCache, op.getResult().getType());
        pLoadOp.mutable_result_type()->set_id(resultTypeID);
        pInst->mutable_load()->CopyFrom(pLoadOp);
      })
      .Case<cir::StoreOp>([instID, pInst, pModuleID, &typeCache,
                           &opCache](cir::StoreOp op) {
        protocir::CIRStoreOp pStoreOp;
        pStoreOp.mutable_base()->set_id(instID);
        auto addressID = internOperation(opCache, op.getAddr().getDefiningOp());
        pStoreOp.mutable_address()->set_id(addressID);
        auto valueID = internOperation(opCache, op.getAddr().getDefiningOp());
        pStoreOp.mutable_value()->set_id(valueID);
        pInst->mutable_store()->CopyFrom(pStoreOp);
      })
      .Case<cir::ConstantOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::ConstantOp op) {
            protocir::CIRConstantOp pConstantOp;
            pConstantOp.mutable_base()->set_id(instID);
            auto resultTypeID = internType(typeCache, op.getResult().getType());
            pConstantOp.mutable_result_type()->set_id(resultTypeID);
            pInst->mutable_constant()->CopyFrom(pConstantOp);
          })
      .Case<cir::CallOp>([instID, pInst, pModuleID, &typeCache, &opCache,
                          &functionCache](cir::CallOp op) {
        protocir::CIRCallOp pCallOp;
        pCallOp.mutable_base()->set_id(instID);
        if (op.getCallee().has_value()) {
          auto callee = op.getCallee().value();
          *pCallOp.mutable_callee() = callee.str();
        }
        for (auto arg : op.getArgOperands()) {
          auto pArg = pCallOp.add_arguments();
          auto argLine = internOperation(opCache, arg.getDefiningOp());
          pArg->set_id(argLine);
        }
        if (op.getNumResults() > 0) {
          auto resultTypeID = internType(typeCache, op.getResult().getType());
          pCallOp.mutable_result_type()->set_id(resultTypeID);
        }
        if (auto callable = op.resolveCallable()) {
          auto callableID =
              internFunction(functionCache, cast<cir::FuncOp>(callable));
        }
        pInst->mutable_call()->CopyFrom(pCallOp);
      })
      .Case<cir::ReturnOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::ReturnOp op) {
            protocir::CIRReturnOp pReturnOp;
            pReturnOp.mutable_base()->set_id(instID);
            for (auto input : op.getInput()) {
              auto inputID = internOperation(opCache, input.getDefiningOp());
              pReturnOp.add_input()->set_id(inputID);
            }
            pInst->mutable_return_()->CopyFrom(pReturnOp);
          })
      .Case<cir::GetGlobalOp>([instID, pInst, pModuleID, &typeCache,
                               &opCache](cir::GetGlobalOp op) {
        protocir::CIRGetGlobalOp pGetGlobalOp;
        pGetGlobalOp.mutable_base()->set_id(instID);
        *pGetGlobalOp.mutable_name() = op.getName().str();
        pInst->mutable_get_global()->CopyFrom(pGetGlobalOp);
      })
      .Case<cir::CastOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::CastOp op) {
            protocir::CIRCastOp pCastOp;
            pCastOp.mutable_base()->set_id(instID);
            auto srcID = internOperation(opCache, op.getSrc().getDefiningOp());
            pCastOp.mutable_src()->set_id(srcID);
            auto resultTypeID = internType(typeCache, op.getResult().getType());
            pCastOp.mutable_result_type()->set_id(resultTypeID);
            pInst->mutable_cast()->CopyFrom(pCastOp);
          })
      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });
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
  TypeCache typeCache;
  FunctionCache functionCache;
  auto &bodyRegion = (*module).getBodyRegion();
  for (auto &bodyBlock : bodyRegion) {
    for (auto &func : bodyBlock) {
      protocir::CIRFunction *pFunction = pModule.add_functions();
      protocir::CIRFunctionID pFunctionID;
      pFunction->mutable_id()->mutable_module_id()->CopyFrom(pModuleID);
      pFunction->mutable_id()->set_id(++func_idx);
      std::string functionName = cast<cir::FuncOp>(func).getSymName().str();
      *pFunction->mutable_name() = functionName;
      OperationCache opCache;
      for (auto &block : cast<cir::FuncOp>(func).getFunctionBody()) {
        for (auto &inst : block) {
          auto pInst = pFunction->add_operations();
          serializeOperation(inst, pInst, pModuleID, typeCache, opCache,
                             functionCache);
        }
      }
    }
  }
  std::string binary;
  pModule.SerializeToString(&binary);
  std::filesystem::path relOutPath = argv[2];

  auto absOutPath = std::filesystem::absolute(relOutPath);
  std::ofstream out(absOutPath.c_str());
  out << binary;
  out.close();
}
