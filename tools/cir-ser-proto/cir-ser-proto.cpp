#include "cir-tac/Serializer.h"

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

using namespace protocir;

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
      std::ignore =
          Serializer::internFunction(functionCache, cast<cir::FuncOp>(func));
    }
  }
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
          std::ignore = Serializer::internOperation(opCache, &inst);
        }
      }
      for (auto &block : cast<cir::FuncOp>(func).getFunctionBody()) {
        for (auto &inst : block) {
          auto pInst = pFunction->add_operations();
          Serializer::serializeOperation(inst, pInst, pModuleID, typeCache,
                                         opCache, functionCache);
        }
      }
      for (auto &type : cast<cir::FuncOp>(func).getArgumentTypes()) {
        auto pType = pFunction->add_arguments_types();
        pType->mutable_module_id()->CopyFrom(pModuleID);
        pType->set_id(Serializer::internType(typeCache, type));
      }
    }
  }
  for (auto &type : typeCache) {
    auto pType = pModule.add_types();
    pType->mutable_id()->set_id(type.getSecond());
    pType->mutable_id()->mutable_module_id()->CopyFrom(pModuleID);
    *pType->mutable_name() = type.getFirst().getAbstractType().getName().str();
  }
  std::string binary;
  pModule.SerializeToString(&binary);
  std::filesystem::path relOutPath = argv[2];

  auto absOutPath = std::filesystem::absolute(relOutPath);
  std::ofstream out(absOutPath.c_str());
  out << binary;
  out.close();
}
