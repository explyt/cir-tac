#pragma once

#include "CXX/CIRFunction.h"

#include <mlir/IR/MLIRContext.h>

#include <filesystem>

class CIRModule;

class CIRReader {
public:
  CIRReader();

  CIRModule loadFromFile(const std::filesystem::path &path);
private:
  mlir::MLIRContext context;
};