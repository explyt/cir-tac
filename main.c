#include "CAPI/CIR.h"
#include "CAPI/CIRFunction.h"
#include "CAPI/CIRInst.h"
#include "CAPI/CIRModule.h"
#include "CAPI/CIRType.h"

#include "CAPI/CIRFunctionAPI.h"
#include "CAPI/CIRInstAPI.h"
#include "CAPI/CIRModuleAPI.h"
#include "CAPI/CIRTypeAPI.h"

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

      // printf("%%%ld = ", inst.idxInOwningFunction);
      switch (inst.opcode) {
      case AllocaOp: {
        printf("alloca %s %ld\n", CIRTypeGetName(CIRAllocaOpType(inst)),
               CIRTypeGetSize(CIRAllocaOpType(inst)));
        break;
      }
      case BinOp:
        printf("binop\n");
        break;
      case StoreOp: {
        printf("store\n");
        break;
      }
      case LoadOp:
        printf("load : %s\n", CIRTypeGetName(CIRLoadOpType(inst)));
        break;
      case CallOp: {
        struct CIRFunctionRef called = CIRCallOpCalledFunction(inst);
        printf("call %s ", CIRFunctionGetName(called));
        printf("(");
        for (size_t argIdx = 0; argIdx < called.argumentsNum; ++argIdx) {
          if (argIdx != 0) {
            printf(", ");
          }
          struct CIRTypeRef argumentTypeRef =
              CIRFunctionGetArgumentType(called, argIdx);
          printf("%s", CIRTypeGetName(argumentTypeRef));
        }
        printf(") -> %s\n", CIRTypeGetName(CIRFunctionGetReturnType(called)));
        break;
      }
      case ConstantOp:
        printf("constant\n");
        break;
      case ReturnOp: {
        struct CIRTypeRef retType = CIRReturnOpGetType(inst);
        printf("return : %s\n", CIRTypeGetName(retType));
        break;
      }
      case UnknownOp:
        printf("unknown\n");
        break;
      }
    }
  }

  CIRDestroyModule(module);
  CIRDestroyReader(reader);
}