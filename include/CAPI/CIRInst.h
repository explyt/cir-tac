#pragma once

#include <stdlib.h>

struct CIRFunctionRef;

enum CIRInstOpcode {
  CirAlloca,
  CirBinop,
  CirLoad,
  CirStore,
  CirReturn,
  CirConst,
};

struct CIRInstRef {
  CIRInstOpcode opcode;
  size_t index;
};

struct CIRInstList {
  size_t size;
};

extern CIRFunctionRef CIRInstGetParentFunction(struct CIRInstRef instRef);
