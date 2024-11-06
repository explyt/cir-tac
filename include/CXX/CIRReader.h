#pragma once

#include "CAPI/CIR.h"
#include "CXX/CIRFunction.h"
#include "CXX/CIRModule.h"

#include <mlir/IR/MLIRContext.h>

#include <filesystem>

class CIRReader {
public:
  CIRReader();

  CIRModule loadFromFile(const std::filesystem::path &path);

  static CIRReader &fromRef(CIRReaderRef reader) {
    return *reinterpret_cast<CIRReader *>(reader.innerRef);
  }

  CIRReaderRef toRef() {
    return CIRReaderRef{reinterpret_cast<uintptr_t>(this)};
  }

private:
  mlir::MLIRContext context;
};
