/* Autogenerated by mlir-tblgen; don't manually edit. */
// clang-format off

#include "cir-tac/TypeSerializer.h"
#include "cir-tac/EnumSerializer.h"
#include "proto/type.pb.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace protocir;

MLIRType TypeSerializer::serializeMLIRType(mlir::Type type) {
  MLIRType pType;

  *pType.mutable_id() = typeCache.getMLIRTypeID(type);

  llvm::TypeSwitch<mlir::Type>(type)
  .Case<mlir::NoneType>([this, &pType](mlir::NoneType type) {
    auto serialized = serializeMLIRNoneType(type);
    *pType.mutable_none_type() = serialized;
  })
  .Case<mlir::IntegerType>([this, &pType](mlir::IntegerType type) {
    auto serialized = serializeMLIRIntegerType(type);
    *pType.mutable_integer_type() = serialized;
  })
  .Case<cir::ArrayType>([this, &pType](cir::ArrayType type) {
    auto serialized = serializeCIRArrayType(type);
    *pType.mutable_array_type() = serialized;
  })
  .Case<cir::BF16Type>([this, &pType](cir::BF16Type type) {
    auto serialized = serializeCIRBFloat16Type(type);
    *pType.mutable_b_float16_type() = serialized;
  })
  .Case<cir::BoolType>([this, &pType](cir::BoolType type) {
    auto serialized = serializeCIRBoolType(type);
    *pType.mutable_bool_type() = serialized;
  })
  .Case<cir::ComplexType>([this, &pType](cir::ComplexType type) {
    auto serialized = serializeCIRComplexType(type);
    *pType.mutable_complex_type() = serialized;
  })
  .Case<cir::DataMemberType>([this, &pType](cir::DataMemberType type) {
    auto serialized = serializeCIRDataMemberType(type);
    *pType.mutable_data_member_type() = serialized;
  })
  .Case<cir::DoubleType>([this, &pType](cir::DoubleType type) {
    auto serialized = serializeCIRDoubleType(type);
    *pType.mutable_double_type() = serialized;
  })
  .Case<cir::ExceptionInfoType>([this, &pType](cir::ExceptionInfoType type) {
    auto serialized = serializeCIRExceptionType(type);
    *pType.mutable_exception_type() = serialized;
  })
  .Case<cir::FP16Type>([this, &pType](cir::FP16Type type) {
    auto serialized = serializeCIRFP16Type(type);
    *pType.mutable_fp16_type() = serialized;
  })
  .Case<cir::FP80Type>([this, &pType](cir::FP80Type type) {
    auto serialized = serializeCIRFP80Type(type);
    *pType.mutable_fp80_type() = serialized;
  })
  .Case<cir::FP128Type>([this, &pType](cir::FP128Type type) {
    auto serialized = serializeCIRFP128Type(type);
    *pType.mutable_fp128_type() = serialized;
  })
  .Case<cir::FuncType>([this, &pType](cir::FuncType type) {
    auto serialized = serializeCIRFuncType(type);
    *pType.mutable_func_type() = serialized;
  })
  .Case<cir::IntType>([this, &pType](cir::IntType type) {
    auto serialized = serializeCIRIntType(type);
    *pType.mutable_int_type() = serialized;
  })
  .Case<cir::LongDoubleType>([this, &pType](cir::LongDoubleType type) {
    auto serialized = serializeCIRLongDoubleType(type);
    *pType.mutable_long_double_type() = serialized;
  })
  .Case<cir::MethodType>([this, &pType](cir::MethodType type) {
    auto serialized = serializeCIRMethodType(type);
    *pType.mutable_method_type() = serialized;
  })
  .Case<cir::PointerType>([this, &pType](cir::PointerType type) {
    auto serialized = serializeCIRPointerType(type);
    *pType.mutable_pointer_type() = serialized;
  })
  .Case<cir::SingleType>([this, &pType](cir::SingleType type) {
    auto serialized = serializeCIRSingleType(type);
    *pType.mutable_single_type() = serialized;
  })
  .Case<cir::VectorType>([this, &pType](cir::VectorType type) {
    auto serialized = serializeCIRVectorType(type);
    *pType.mutable_vector_type() = serialized;
  })
  .Case<cir::VoidType>([this, &pType](cir::VoidType type) {
    auto serialized = serializeCIRVoidType(type);
    *pType.mutable_void_type() = serialized;
  })
  .Case<cir::StructType>([this, &pType](cir::StructType type) {
    auto serialized = serializeCIRStructType(type);
    *pType.mutable_struct_type() = serialized;
  })
  .Default([](mlir::Type type) {
    type.dump();
    llvm_unreachable("unknown type during serialization");
  });

  return pType;
}

MLIRNoneType TypeSerializer::serializeMLIRNoneType(mlir::NoneType type) {
  MLIRNoneType serialized;
  return serialized;
}

MLIRIntegerType TypeSerializer::serializeMLIRIntegerType(mlir::IntegerType type) {
  MLIRIntegerType serialized;
  serialized.set_width(type.getWidth());
  serialized.set_signed_(type.isSigned());
  return serialized;
}

CIRArrayType TypeSerializer::serializeCIRArrayType(cir::ArrayType type) {
  CIRArrayType serialized;
  *serialized.mutable_elt_type() = typeCache.getMLIRTypeID(type.getEltType());
  serialized.set_size(type.getSize());
  return serialized;
}

CIRBFloat16Type TypeSerializer::serializeCIRBFloat16Type(cir::BF16Type type) {
  CIRBFloat16Type serialized;
  return serialized;
}

CIRBoolType TypeSerializer::serializeCIRBoolType(cir::BoolType type) {
  CIRBoolType serialized;
  return serialized;
}

CIRComplexType TypeSerializer::serializeCIRComplexType(cir::ComplexType type) {
  CIRComplexType serialized;
  *serialized.mutable_element_ty() = typeCache.getMLIRTypeID(type.getElementTy());
  return serialized;
}

CIRDataMemberType TypeSerializer::serializeCIRDataMemberType(cir::DataMemberType type) {
  CIRDataMemberType serialized;
  *serialized.mutable_member_ty() = typeCache.getMLIRTypeID(type.getMemberTy());
  *serialized.mutable_cls_ty() = typeCache.getMLIRTypeID(type.getClsTy());
  return serialized;
}

CIRDoubleType TypeSerializer::serializeCIRDoubleType(cir::DoubleType type) {
  CIRDoubleType serialized;
  return serialized;
}

CIRExceptionType TypeSerializer::serializeCIRExceptionType(cir::ExceptionInfoType type) {
  CIRExceptionType serialized;
  return serialized;
}

CIRFP16Type TypeSerializer::serializeCIRFP16Type(cir::FP16Type type) {
  CIRFP16Type serialized;
  return serialized;
}

CIRFP80Type TypeSerializer::serializeCIRFP80Type(cir::FP80Type type) {
  CIRFP80Type serialized;
  return serialized;
}

CIRFP128Type TypeSerializer::serializeCIRFP128Type(cir::FP128Type type) {
  CIRFP128Type serialized;
  return serialized;
}

CIRFuncType TypeSerializer::serializeCIRFuncType(cir::FuncType type) {
  CIRFuncType serialized;
  for (auto i : type.getInputs()) {
    serialized.mutable_inputs()->Add(typeCache.getMLIRTypeID(i));
  }
  *serialized.mutable_return_type() = typeCache.getMLIRTypeID(type.getReturnType());
  serialized.set_var_arg(type.getVarArg());
  return serialized;
}

CIRIntType TypeSerializer::serializeCIRIntType(cir::IntType type) {
  CIRIntType serialized;
  serialized.set_width(type.getWidth());
  serialized.set_is_signed(type.getIsSigned());
  return serialized;
}

CIRLongDoubleType TypeSerializer::serializeCIRLongDoubleType(cir::LongDoubleType type) {
  CIRLongDoubleType serialized;
  *serialized.mutable_underlying() = typeCache.getMLIRTypeID(type.getUnderlying());
  return serialized;
}

CIRMethodType TypeSerializer::serializeCIRMethodType(cir::MethodType type) {
  CIRMethodType serialized;
  *serialized.mutable_member_func_ty() = typeCache.getMLIRTypeID(type.getMemberFuncTy());
  *serialized.mutable_cls_ty() = typeCache.getMLIRTypeID(type.getClsTy());
  return serialized;
}

CIRPointerType TypeSerializer::serializeCIRPointerType(cir::PointerType type) {
  CIRPointerType serialized;
  *serialized.mutable_pointee() = typeCache.getMLIRTypeID(type.getPointee());
  if (type.getAddrSpace()) {
    *serialized.mutable_addr_space() = attributeSerializer.serializeMLIRAttribute(type.getAddrSpace());
  }
  return serialized;
}

CIRSingleType TypeSerializer::serializeCIRSingleType(cir::SingleType type) {
  CIRSingleType serialized;
  return serialized;
}

CIRVectorType TypeSerializer::serializeCIRVectorType(cir::VectorType type) {
  CIRVectorType serialized;
  *serialized.mutable_elt_type() = typeCache.getMLIRTypeID(type.getEltType());
  serialized.set_size(type.getSize());
  return serialized;
}

CIRVoidType TypeSerializer::serializeCIRVoidType(cir::VoidType type) {
  CIRVoidType serialized;
  return serialized;
}

CIRStructType TypeSerializer::serializeCIRStructType(cir::StructType type) {
 CIRStructType serialized;
 for (auto i : type.getMembers()) {
   serialized.mutable_members()->Add(typeCache.getMLIRTypeID(i));
 }
 *serialized.mutable_name() = attributeSerializer.serializeMLIRStringAttr(type.getName());
 serialized.set_incomplete(type.getIncomplete());
 serialized.set_packed(type.getPacked());
 serialized.set_kind(serializeCIRRecordKind(type.getKind()));
 return serialized;
}

// clang-format on
