#include "CXX/CIRModule.h"

#include "CAPI/CIRModule.h"
#include "CAPI/CIRModuleAPI.h"

#include <cinttypes>

CIRFunctionRef CIRModuleGetFunction(struct CIRModuleRef moduleRef, size_t idx) {
  auto &module = CIRModule::fromRef(moduleRef);
  const auto &cirFunction = module.functionsList()[idx];
  return cirFunction.toRef();
}
