#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CIRInstOpCode.h"
#include "CXX/CIRFunction.h"
#include "CXX/CIRInst.h"
#include <cassert>
#include <sys/_types/_uintptr_t.h>

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
