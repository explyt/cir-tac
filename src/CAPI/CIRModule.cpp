#include "CAPI/CIRModule.h"
#include "CXX/CIRModule.h"
#include <sys/_types/_uintptr_t.h>

CIRFunctionRef CIRModuleGetFunction(struct CIRModuleRef moduleRef, size_t idx) {
  auto &module = CIRModule::fromRef(moduleRef);
  const auto &cirFunction = module.functionsList()[idx];
  return cirFunction.toRef();
}
