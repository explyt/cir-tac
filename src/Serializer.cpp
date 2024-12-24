#include "cir-tac/Serializer.h"
#include "proto/setup.pb.h"
#include "proto/type.pb.h"

#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Value.h>

using namespace protocir;

protocir::CIRValue Serializer::serializeValue(mlir::Value &mlirValue,
                                              protocir::CIRModuleID pModuleID,
                                              TypeCache &typeCache,
                                              OperationCache &opCache,
                                              BlockCache &blockCache) {
  protocir::CIRValue pValue;

  llvm::TypeSwitch<mlir::Value>(mlirValue)
      .Case<mlir::OpResult>(
          [&pValue, &opCache, &blockCache](mlir::OpResult value) {
            protocir::CIROpResult opResult;
            auto op = value.getOwner();
            auto opID = Serializer::internOperation(opCache, op);
            protocir::CIROpID popID;
            popID.set_id(opID);
            *opResult.mutable_owner() = popID;
            auto resultNumber = value.getResultNumber();
            opResult.set_result_number(resultNumber);
            pValue.mutable_op_result()->CopyFrom(opResult);
          })
      .Case<mlir::BlockArgument>(
          [&pValue, &opCache, &blockCache](mlir::BlockArgument value) {
            protocir::CIRBlockArgument blockArgument;
            auto block = value.getOwner();
            auto blockID = Serializer::internBlock(blockCache, block);
            protocir::CIRBlockID pblockID;
            pblockID.set_id(blockID);
            *blockArgument.mutable_owner() = pblockID;
            auto argNumber = value.getArgNumber();
            blockArgument.set_arg_number(argNumber);
            pValue.mutable_block_argument()->CopyFrom(blockArgument);
          })
      .Default([](mlir::Value value) {
        value.dump();
        llvm_unreachable("NIY");
      });
  return pValue;
}

protocir::CIRType Serializer::serializeType(::mlir::Type &cirType,
                                            TypeCache &typeCache) {
  protocir::CIRType pType;
  std::string nameStr;
  llvm::raw_string_ostream nameStream(nameStr);
  cirType.print(nameStream);
  *pType.mutable_id()->mutable_id() = nameStr;
  llvm::TypeSwitch<mlir::Type>(cirType)
      .Case<cir::IntType>([&pType](cir::IntType type) {
        protocir::CIRIntType intType;
        intType.set_width(type.getWidth());
        intType.set_is_signed(type.getIsSigned());

        pType.mutable_int_type()->CopyFrom(intType);
      })
      .Case<cir::SingleType>([&pType](cir::SingleType type) {
        protocir::CIRSingleType singleType;
        pType.mutable_single_type()->CopyFrom(singleType);
      })
      .Case<cir::DoubleType>([&pType](cir::DoubleType type) {
        protocir::CIRDoubleType doubleType;
        pType.mutable_double_type()->CopyFrom(doubleType);
      })
      .Case<cir::FP16Type>([&pType](cir::FP16Type type) {
        protocir::CIRFP16Type fp16Type;
        pType.mutable_fp16_type()->CopyFrom(fp16Type);
      })
      .Case<cir::BF16Type>([&pType](cir::BF16Type type) {
        protocir::CIRBF16Type bf16Type;
        pType.mutable_bf16_type()->CopyFrom(bf16Type);
      })
      .Case<cir::FP80Type>([&pType](cir::FP80Type type) {
        protocir::CIRFP80Type fp80Type;
        pType.mutable_fp80_type()->CopyFrom(fp80Type);
      })
      .Case<cir::FP128Type>([&pType](cir::FP128Type type) {
        protocir::CIRFP128Type fp128Type;
        pType.mutable_fp128_type()->CopyFrom(fp128Type);
      })
      .Case<cir::LongDoubleType>([&pType](cir::LongDoubleType type) {
        protocir::CIRLongDoubleType longDoubleType;
        pType.mutable_long_double_type()->CopyFrom(longDoubleType);
      })
      .Case<cir::ComplexType>([&pType](cir::ComplexType type) {
        protocir::CIRComplexType complexType;
        pType.mutable_complex_type()->CopyFrom(complexType);
      })
      .Case<cir::PointerType>([&pType](cir::PointerType type) {
        protocir::CIRPointerType pointerType;
        pType.mutable_pointer_type()->CopyFrom(pointerType);
      })
      .Case<cir::DataMemberType>([&pType](cir::DataMemberType type) {
        protocir::CIRDataMemberType dataMemberType;
        pType.mutable_data_member_type()->CopyFrom(dataMemberType);
      })
      .Case<cir::BoolType>([&pType](cir::BoolType type) {
        protocir::CIRBoolType boolType;
        pType.mutable_bool_type()->CopyFrom(boolType);
      })
      .Case<cir::ArrayType>([&pType](cir::ArrayType type) {
        protocir::CIRArrayType arrayType;
        pType.mutable_array_type()->CopyFrom(arrayType);
      })
      .Case<cir::VectorType>([&pType](cir::VectorType type) {
        protocir::CIRVectorType vectorType;
        pType.mutable_vector_type()->CopyFrom(vectorType);
      })
      .Case<cir::FuncType>([&pType](cir::FuncType type) {
        protocir::CIRFuncType funcType;
        pType.mutable_func_type()->CopyFrom(funcType);
      })
      .Case<cir::MethodType>([&pType](cir::MethodType type) {
        protocir::CIRMethodType methodType;
        pType.mutable_method_type()->CopyFrom(methodType);
      })
      .Case<cir::ExceptionInfoType>([&pType](cir::ExceptionInfoType type) {
        protocir::CIRExceptionInfoType exceptionInfoType;
        pType.mutable_exception_info_type()->CopyFrom(exceptionInfoType);
      })
      .Case<cir::VoidType>([&pType](cir::VoidType type) {
        protocir::CIRVoidType voidType;
        pType.mutable_void_type()->CopyFrom(voidType);
      })
      .Default([](mlir::Type type) {
        type.dump();
        llvm_unreachable("NIY");
      });
  return pType;
}
