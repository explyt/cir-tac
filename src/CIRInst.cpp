#include "CXX/CIRInst.h"
#include "CIRInstOpCode.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#define TRY_RESOLVE_TO(ID, OP_KIND)                                            \
  do {                                                                         \
    if (ID == mlir::TypeID::get<mlir::cir::OP_KIND>()) {                       \
      return CIROpCode::OP_KIND;                                               \
    }                                                                          \
  } while (false)

CIROpCode CIRInst::opcode() const {
  auto opTypeID = inst.getName().getTypeID();

  // FIXME: Do switch : case
  TRY_RESOLVE_TO(opTypeID, AllocaOp);
  TRY_RESOLVE_TO(opTypeID, BinOp);
  TRY_RESOLVE_TO(opTypeID, LoadOp);
  TRY_RESOLVE_TO(opTypeID, StoreOp);
  TRY_RESOLVE_TO(opTypeID, ConstantOp);
  TRY_RESOLVE_TO(opTypeID, CallOp);
  TRY_RESOLVE_TO(opTypeID, ReturnOp);

  return CIROpCode::UnknownOp;
}
#undef TRY_RESOLVE_TO
