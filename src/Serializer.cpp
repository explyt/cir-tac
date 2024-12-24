#include "cir-tac/Serializer.h"
#include "proto/setup.pb.h"
#include "proto/type.pb.h"

#include <clang/CIR/Dialect/IR/CIRAttrs.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
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

protocir::CIRRecordKind
Serializer::serializeRecordKind(::cir::StructType::RecordKind cirKind) {
  switch (cirKind) {
  case cir::StructType::Class:
    return protocir::CIRRecordKind::RecordKind_Class;
  case cir::StructType::Union:
    return protocir::CIRRecordKind::RecordKind_Union;
  case cir::StructType::Struct:
    return protocir::CIRRecordKind::RecordKind_Struct;
  }
}

protocir::CIRType Serializer::serializeType(::mlir::Type &cirType,
                                            protocir::CIRModuleID pModuleID,
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
      .Case<cir::LongDoubleType>(
          [&pType, &typeCache, pModuleID](cir::LongDoubleType type) {
            protocir::CIRLongDoubleType longDoubleType;
            auto underlying = type.getUnderlying();
            auto underlyingID = internType(typeCache, underlying);
            protocir::CIRTypeID punderlying;
            *punderlying.mutable_module_id() = pModuleID;
            *punderlying.mutable_id() = underlyingID;
            *longDoubleType.mutable_underlying() = punderlying;
            pType.mutable_long_double_type()->CopyFrom(longDoubleType);
          })
      .Case<cir::ComplexType>(
          [&pType, &typeCache, pModuleID](cir::ComplexType type) {
            protocir::CIRComplexType complexType;
            auto elementTy = type.getElementTy();
            auto elementTyID = internType(typeCache, elementTy);
            protocir::CIRTypeID pelementTy;
            *pelementTy.mutable_module_id() = pModuleID;
            *pelementTy.mutable_id() = elementTyID;
            *complexType.mutable_element_ty() = pelementTy;
            pType.mutable_complex_type()->CopyFrom(complexType);
          })
      .Case<cir::PointerType>([&pType, &typeCache,
                               pModuleID](cir::PointerType type) {
        protocir::CIRPointerType pointerType;
        auto pointee = type.getPointee();
        auto pointeeID = internType(typeCache, pointee);
        protocir::CIRTypeID ppointee;
        *ppointee.mutable_module_id() = pModuleID;
        *ppointee.mutable_id() = pointeeID;
        *pointerType.mutable_pointee() = ppointee;
        auto addrSpaceOptional = type.getAddrSpace();
        if (addrSpaceOptional) {
          auto addrSpace = llvm::cast<cir::AddressSpaceAttr>(addrSpaceOptional);
          pointerType.set_addr_space(addrSpace.getValue());
        }

        pType.mutable_pointer_type()->CopyFrom(pointerType);
      })
      .Case<cir::DataMemberType>(
          [&pType, &typeCache, pModuleID](cir::DataMemberType type) {
            protocir::CIRDataMemberType dataMemberType;
            auto memberTy = type.getMemberTy();
            auto memberTyID = internType(typeCache, memberTy);
            protocir::CIRTypeID pmemberTy;
            *pmemberTy.mutable_module_id() = pModuleID;
            *pmemberTy.mutable_id() = memberTyID;
            *dataMemberType.mutable_member_ty() = pmemberTy;
            auto clsTy = type.getClsTy();
            auto clsTyID = internType(typeCache, clsTy);
            protocir::CIRTypeID pclsTy;
            *pclsTy.mutable_module_id() = pModuleID;
            *pclsTy.mutable_id() = clsTyID;
            *dataMemberType.mutable_cls_ty() = pclsTy;
            pType.mutable_data_member_type()->CopyFrom(dataMemberType);
          })
      .Case<cir::BoolType>([&pType](cir::BoolType type) {
        protocir::CIRBoolType boolType;
        pType.mutable_bool_type()->CopyFrom(boolType);
      })
      .Case<cir::ArrayType>(
          [&pType, &typeCache, pModuleID](cir::ArrayType type) {
            protocir::CIRArrayType arrayType;
            auto eltTy = type.getEltType();
            auto eltTyID = internType(typeCache, eltTy);
            protocir::CIRTypeID peltTy;
            *peltTy.mutable_module_id() = pModuleID;
            *peltTy.mutable_id() = eltTyID;
            *arrayType.mutable_elt_ty() = peltTy;
            arrayType.set_size(type.getSize());
            pType.mutable_array_type()->CopyFrom(arrayType);
          })
      .Case<cir::VectorType>(
          [&pType, &typeCache, pModuleID](cir::VectorType type) {
            protocir::CIRVectorType vectorType;
            auto eltTy = type.getEltType();
            auto eltTyID = internType(typeCache, eltTy);
            protocir::CIRTypeID peltTy;
            *peltTy.mutable_module_id() = pModuleID;
            *peltTy.mutable_id() = eltTyID;
            *vectorType.mutable_elt_ty() = peltTy;
            vectorType.set_size(type.getSize());
            pType.mutable_vector_type()->CopyFrom(vectorType);
          })
      .Case<cir::FuncType>([&pType, &typeCache, pModuleID](cir::FuncType type) {
        protocir::CIRFuncType funcType;
        for (auto &input : type.getInputs()) {
          auto inputID = internType(typeCache, input);
          protocir::CIRTypeID pinputTy;
          *pinputTy.mutable_module_id() = pModuleID;
          *pinputTy.mutable_id() = inputID;
          funcType.add_inputs()->CopyFrom(pinputTy);
        }
        auto returnType = type.getReturnType();
        auto returnTypeID = internType(typeCache, returnType);
        protocir::CIRTypeID preturnType;
        *preturnType.mutable_module_id() = pModuleID;
        *preturnType.mutable_id() = returnTypeID;
        *funcType.mutable_return_type() = preturnType;
        funcType.set_var_arg(type.getVarArg());
        pType.mutable_func_type()->CopyFrom(funcType);
      })
      .Case<cir::MethodType>(
          [&pType, &typeCache, pModuleID](cir::MethodType type) {
            protocir::CIRMethodType methodType;
            auto memberFuncTy = type.getMemberFuncTy();
            auto memberFuncTyID = internType(typeCache, memberFuncTy);
            protocir::CIRTypeID pmemberFuncTy;
            *pmemberFuncTy.mutable_module_id() = pModuleID;
            *pmemberFuncTy.mutable_id() = memberFuncTyID;
            *methodType.mutable_member_func_ty() = pmemberFuncTy;
            auto clsTy = type.getClsTy();
            auto clsTyID = internType(typeCache, clsTy);
            protocir::CIRTypeID pclsTy;
            *pclsTy.mutable_module_id() = pModuleID;
            *pclsTy.mutable_id() = clsTyID;
            *methodType.mutable_cls_ty() = pclsTy;
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
      .Case<cir::StructType>([&pType, &typeCache,
                              pModuleID](cir::StructType type) {
        protocir::CIRStructType structType;
        *structType.mutable_name() = type.getName().str();
        structType.set_kind(Serializer::serializeRecordKind(type.getKind()));
        structType.set_packed(type.getPacked());
        for (auto &member : type.getMembers()) {
          auto memberID = internType(typeCache, member);
          protocir::CIRTypeID pmember;
          *pmember.mutable_module_id() = pModuleID;
          *pmember.mutable_id() = memberID;
          structType.add_members()->CopyFrom(pmember);
        }
        pType.mutable_struct_type()->CopyFrom(structType);
      })
      .Default([](mlir::Type type) {
        type.dump();
        llvm_unreachable("NIY");
      });
  return pType;
}
