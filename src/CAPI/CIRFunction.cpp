#include "CAPI/CIRFunction.h"
#include "CAPI/CIRFunctionAPI.h"
#include "CAPI/CIRInst.h"

#include "CIRInstOpCode.h"
#include "CXX/CIRFunction.h"
#include "CXX/CIRInst.h"

#include <cassert>
#include <cinttypes>

struct CIRInstRef CIRFunctionGetInst(struct CIRFunctionRef funcRef,
                                     size_t idx) {
  assert(idx < funcRef.instructionsNum);

  auto function = CIRFunction::fromRef(funcRef);
  auto &inst = function.instructionsList()[idx];

  return inst.toRef();
}

const char *CIRFunctionGetName(struct CIRFunctionRef funcRef) {
  auto function = CIRFunction::fromRef(funcRef);
  return function.getName();
}

CIRTypeRef CIRFunctionGetReturnType(CIRFunctionRef funcRef) {
  auto function = CIRFunction::fromRef(funcRef);
  return function.getReturnType().toRef();
}

struct CIRTypeRef CIRFunctionGetArgumentType(struct CIRFunctionRef funcRef,
                                             size_t argNum) {
  auto function = CIRFunction::fromRef(funcRef);
  return function.getArgumentType(argNum).toRef();
}
