#include "CXX/CIRReader.h"
#include "CXX/CIRModule.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

CIRReader::CIRReader() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::cir::CIRDialect>();

  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects();
}

CIRModule CIRReader::loadFromFile(const std::filesystem::path &path) {
  auto absPath = std::filesystem::absolute(path);
  if (!std::filesystem::exists(absPath)) {
    throw std::runtime_error("missing path given");
  }

  mlir::ParserConfig parseConfig(&context);
  return CIRModule(
      mlir::parseSourceFile<mlir::ModuleOp>(path.c_str(), parseConfig));
}
