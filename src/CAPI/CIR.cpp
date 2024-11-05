#include "CAPI/CIR.h"
#include "CXX/CIRModule.h"
#include "CXX/CIRReader.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/CAPI/Registration.h>
#include <sys/_types/_uintptr_t.h>

CIRReaderRef CIRCreateReader() {
  auto reader = new CIRReader();
  return reader->toRef();
}

void CIRDestroyReader(CIRReaderRef readerRef) {
  delete &CIRReader::fromRef(readerRef);
}

CIRModuleRef loadModuleFromFile(CIRReaderRef readerRef, const char *name) {
  auto reader = reinterpret_cast<CIRReader *>(readerRef.innerRef);

  try {
    auto module = new CIRModule(reader->loadFromFile(name));
    return module->toRef();
  } catch (...) {
    return CIRModuleRef{reinterpret_cast<uintptr_t>(nullptr)};
  }
}

void CIRDestroyModule(CIRModuleRef moduleRef) {
  delete &CIRModule::fromRef(moduleRef);
}
