#include "CAPI/CIR.h"
#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"
#include "CAPI/CIRType.h"

#include <assert.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  struct CIRReaderRef reader = CIRCreateReader();
  struct CIRModuleRef module = loadModuleFromFile(reader, argv[1]);

  assert(module.innerRef != 0);
  size_t functionNum = module.functionsNum;

  for (size_t funcIdx = 0; funcIdx < functionNum; ++funcIdx) {
    struct CIRFunctionRef function = CIRModuleGetFunction(module, funcIdx);
    for (size_t instIdx = 0; instIdx < function.instructionsNum; ++instIdx) {
      struct CIRInstRef inst = CIRFunctionGetInst(function, instIdx);

      switch (inst.opcode) {
      case AllocaOp: {
        printf("alloca %s\n", CIRTypeGetName(CIRAllocaOpType(inst)));
        break;
      }
      case BinOp:
        printf("binop\n");
        break;
      case StoreOp:
        printf("store\n");
        break;
      case LoadOp:
        printf("load\n");
        break;
      case CallOp: {
        struct CIRFunctionRef called = CIRCallOpCalledFunction(inst);
        printf("call %s\n", CIRFunctionGetName(called));
        break;
      }
      case ConstantOp:
        printf("constant\n");
        break;
      case ReturnOp:
        printf("return\n");
        break;
      case UnknownOp:
        printf("unknown\n");
        break;
      }
    }
  }

  CIRDestroyReader(reader);
}