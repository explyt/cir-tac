#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum CIROpCode {
  AllocaOp,
  BinOp,
  StoreOp,
  LoadOp,
  CallOp,
  ConstantOp,
  ReturnOp,

  UnknownOp,
};

#ifdef __cplusplus
}
#endif
