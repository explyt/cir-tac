#include "CAPI/CIR.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/CAPI/Registration.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CIR, cir, mlir::cir::CIRDialect)

#define TRY_RESOLVE_TO(ID, OP_KIND)                                            \
  do {                                                                         \
    if (ID == mlir::TypeID::get<mlir::cir::OP_KIND>()) {                       \
      return MlirCIROpCode::OP_KIND;                                           \
    }                                                                          \
  } while (false)

MlirCIROpCode mlirCIRdetermineOpCode(MlirOperation op) {
  auto operation = unwrap(op);
  auto opTypeID = operation->getName().getTypeID();

  TRY_RESOLVE_TO(opTypeID, AllocaOp);
  TRY_RESOLVE_TO(opTypeID, BinOp);
  TRY_RESOLVE_TO(opTypeID, LoadOp);
  TRY_RESOLVE_TO(opTypeID, StoreOp);
  TRY_RESOLVE_TO(opTypeID, ConstantOp);
  TRY_RESOLVE_TO(opTypeID, CallOp);
  TRY_RESOLVE_TO(opTypeID, ReturnOp);

  return MlirCIROpCode::UnknownOp;
}

#undef TRY_RESOLVE_TO
