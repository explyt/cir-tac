#include "cir-tac/Deserializer.h"
#include "cir-tac/AttrDeserializer.h"
#include "cir-tac/EnumDeserializer.h"
#include "cir-tac/OpDeserializer.h"
#include "cir-tac/Util.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "proto/setup.pb.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <mlir/IR/Verifier.h>
#include <stdexcept>

using namespace protocir;

mlir::Type Deserializer::getType(ModuleInfo &mInfo, const MLIRTypeID &typeId) {
  auto typeIdStr = typeId.id();
  llvm::errs() << "getting type " << typeIdStr << "\n\n";
  if (mInfo.types.find(typeIdStr) == mInfo.types.end()) {
    defineType(mInfo, mInfo.serTypes.at(typeIdStr));
  }
  auto rTy = mInfo.types.at(typeIdStr);
  return rTy;
}

void Deserializer::defineType(ModuleInfo &mInfo, const MLIRType &pTy) {
  llvm::errs() << "defining " << pTy.type_case() << "\n\n";
  auto ctx = &mInfo.ctx;
  mlir::Type rTy;
  switch (pTy.type_case()) {
  case MLIRType::TypeCase::kCirIntType: {
    auto *nctx = new mlir::MLIRContext();
    auto ty = pTy.cir_int_type();
    llvm::errs() << "width: " << ty.width() << " sign: " << ty.is_signed() << " ctx: " << ctx << "\n\n";
    rTy = cir::IntType::get(ctx, ty.width(), ty.is_signed());
    llvm::errs() << "huh";
  } break;
  case MLIRType::TypeCase::kMlirIntegerType: {
    auto ty = pTy.mlir_integer_type();
    auto signedness =
        EnumDeserializer::deserializeMLIRSignednessSemantics(ty.signedness());
    rTy = mlir::IntegerType::get(ctx, ty.width(), signedness);
  } break;
  case MLIRType::TypeCase::kCirSingleType:
    rTy = cir::SingleType::get(ctx);
    break;
  case MLIRType::TypeCase::kCirDoubleType:
    rTy = cir::DoubleType::get(ctx);
    break;
  case MLIRType::TypeCase::kCirfp16Type:
    rTy = cir::FP16Type::get(ctx);
    break;
  case MLIRType::TypeCase::kCirbFloat16Type:
    rTy = cir::BF16Type::get(ctx);
    break;
  case MLIRType::TypeCase::kCirfp80Type:
    rTy = cir::FP80Type::get(ctx);
    break;
  case MLIRType::TypeCase::kCirfp128Type:
    rTy = cir::FP128Type::get(ctx);
    break;
  case MLIRType::TypeCase::kCirLongDoubleType: {
    auto underlyingTy = getType(mInfo, pTy.cir_long_double_type().underlying());
    rTy = cir::LongDoubleType::get(ctx, underlyingTy);
  } break;
  case MLIRType::TypeCase::kCirComplexType: {
    auto elementTy = getType(mInfo, pTy.cir_complex_type().element_ty());
    rTy = cir::ComplexType::get(ctx, elementTy);
  } break;
  case MLIRType::TypeCase::kCirPointerType: {
    auto pointee = getType(mInfo, pTy.cir_pointer_type().pointee());
    if (pTy.cir_pointer_type().has_addr_space()) {
      auto addrSpace = pTy.cir_pointer_type().addr_space();
      assert(0 && "needs attr deserializer!!!");
    } else {
      rTy = cir::PointerType::get(ctx, pointee);
    }
  } break;
  case MLIRType::TypeCase::kCirDataMemberType: {
    auto memTy = getType(mInfo, pTy.cir_data_member_type().member_ty());
    auto clsTy = getType(mInfo, pTy.cir_data_member_type().cls_ty());
    assert((mlir::isa<cir::StructType>(clsTy)) &&
           "clsTy should be a StructType!");
    rTy = cir::DataMemberType::get(ctx, memTy,
                                   mlir::cast<cir::StructType>(clsTy));
  } break;
  case MLIRType::TypeCase::kCirBoolType:
    rTy = cir::BoolType::get(ctx);
    break;
  case MLIRType::TypeCase::kCirArrayType: {
    auto elTy = getType(mInfo, pTy.cir_array_type().elt_type());
    auto size = pTy.cir_array_type().size();
    rTy = cir::ArrayType::get(ctx, elTy, size);
  } break;
  case MLIRType::TypeCase::kMlirVectorType: {
    auto elTy = getType(mInfo, pTy.mlir_vector_type().element_type());
    std::vector<long long> shape_sizes;
    for (int i = 0; i < pTy.mlir_vector_type().shape_size(); i++) {
      shape_sizes.emplace_back(pTy.mlir_vector_type().shape(i));
    }
    bool *scalable_dims = new bool[1024]();
    auto dims_size = pTy.mlir_vector_type().scalable_dims_size();
    if (dims_size >= 1024) {
      throw std::runtime_error("Scalable dims of mlir::VectorType is too large!");
    }
    for (int i = 0; i < dims_size; i++) {
      scalable_dims[i] = pTy.mlir_vector_type().scalable_dims(i);
    }
    rTy = mlir::VectorType::get(shape_sizes, elTy, llvm::ArrayRef<bool>(scalable_dims, dims_size));
    delete[] scalable_dims;
  } break;
  case MLIRType::TypeCase::kCirVectorType: {
    auto elTy = getType(mInfo, pTy.cir_vector_type().elt_type());
    auto size = pTy.cir_vector_type().size();
    rTy = cir::VectorType::get(ctx, elTy, size);
  } break;
  case MLIRType::TypeCase::kCirFuncType: {
    std::vector<mlir::Type> vecInputTys;
    for (auto ty : pTy.cir_func_type().inputs())
      vecInputTys.push_back(getType(mInfo, ty));
    auto inputTys = mlir::ArrayRef<mlir::Type>(vecInputTys);
    auto returnTy = getType(mInfo, pTy.cir_func_type().return_type());
    auto isVarArg = pTy.cir_func_type().var_arg();
    rTy = cir::FuncType::get(ctx, inputTys, returnTy, isVarArg);
  } break;
  case MLIRType::TypeCase::kCirMethodType: {
    auto memTy = getType(mInfo, pTy.cir_method_type().member_func_ty());
    assert((mlir::isa<cir::FuncType>(memTy)) &&
           "memberFuncTy should be a FuncType!");
    auto clsTy = getType(mInfo, pTy.cir_method_type().cls_ty());
    assert((mlir::isa<cir::StructType>(clsTy)) &&
           "clsTy should be a StructType!");
    rTy = cir::MethodType::get(ctx, mlir::cast<cir::FuncType>(memTy),
                               mlir::cast<cir::StructType>(clsTy));
  } break;
  case MLIRType::TypeCase::kMlirbFloat16Type:
    rTy = mlir::Float16Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirComplexType: {
    auto elTy = getType(mInfo, pTy.mlir_complex_type().element_type());
    rTy = mlir::ComplexType::get(elTy);
  } break;
  case MLIRType::TypeCase::kMlirFloat4E2M1FnType:
    rTy = mlir::Float4E2M1FNType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat6E2M3FnType:
    rTy = mlir::Float6E2M3FNType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat6E3M2FnType:
    rTy = mlir::Float6E3M2FNType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E3M4Type:
    rTy = mlir::Float8E3M4Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E4M3Type:
    rTy = mlir::Float8E4M3Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E4M3B11FnuzType:
    rTy = mlir::Float8E4M3B11FNUZType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E4M3FnType:
    rTy = mlir::Float8E4M3FNType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E4M3FnuzType:
    rTy = mlir::Float8E4M3B11FNUZType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E5M2Type:
    rTy = mlir::Float8E5M2Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E5M2FnuzType:
    rTy = mlir::Float8E4M3B11FNUZType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat8E8M0FnuType:
    rTy = mlir::Float8E8M0FNUType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat16Type:
    rTy = mlir::Float16Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat32Type:
    rTy = mlir::Float32Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat64Type:
    rTy = mlir::Float64Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat80Type:
    rTy = mlir::Float80Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloat128Type:
    rTy = mlir::Float128Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFloatTf32Type:
    rTy = mlir::FloatTF32Type::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirFunctionType: {
    std::vector<mlir::Type> inputs;
    for (int i = 0; i < pTy.mlir_function_type().inputs_size(); i++) {
      inputs.emplace_back(getType(mInfo, pTy.mlir_function_type().inputs(i)));
    }
    std::vector<mlir::Type> results;
    for (int i = 0; i < pTy.mlir_function_type().results_size(); i++) {
      results.emplace_back(getType(mInfo, pTy.mlir_function_type().results(i)));
    }
    rTy = mlir::FunctionType::get(ctx, inputs, results);
  } break;
  case MLIRType::TypeCase::kMlirIndexType:
    rTy = mlir::IndexType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirMemRefType: {
    std::vector<long long> shapes;
    for (int i = 0; i < pTy.mlir_mem_ref_type().shape_size(); i++) {
      shapes.emplace_back(pTy.mlir_mem_ref_type().shape(i));
    }
    auto elTy = getType(mInfo, pTy.mlir_mem_ref_type().element_type());
    auto layout = AttrDeserializer::deserializeMLIRAttribute(
        mInfo, pTy.mlir_mem_ref_type().layout());
    auto memory_space = AttrDeserializer::deserializeMLIRAttribute(
        mInfo, pTy.mlir_mem_ref_type().memory_space());
    rTy = mlir::MemRefType::get(
        shapes, elTy, mlir::cast<mlir::MemRefLayoutAttrInterface>(layout),
        memory_space);
  } break;
  case MLIRType::TypeCase::kMlirNoneType:
    rTy = mlir::NoneType::get(ctx);
    break;
  case MLIRType::TypeCase::kMlirOpaqueType: {
    auto dialect_namespace = AttrDeserializer::deserializeMLIRStringAttr(
        mInfo, pTy.mlir_opaque_type().dialect_namespace());
    auto type_data = pTy.mlir_opaque_type().type_data();
    rTy = mlir::OpaqueType::get(dialect_namespace, type_data);
  } break;
  case MLIRType::TypeCase::kMlirRankedTensorType: {
    std::vector<long long> shapes;
    for (int i = 0; i < pTy.mlir_ranked_tensor_type().shape_size(); i++) {
      shapes.emplace_back(pTy.mlir_ranked_tensor_type().shape(i));
    }
    auto elTy = getType(mInfo, pTy.mlir_ranked_tensor_type().element_type());
    auto encoding = AttrDeserializer::deserializeMLIRAttribute(
        mInfo, pTy.mlir_ranked_tensor_type().encoding());
    rTy = mlir::RankedTensorType::get(shapes, elTy, encoding);
  } break;
  case MLIRType::TypeCase::kMlirTupleType: {
    std::vector<mlir::Type> types;
    for (int i = 0; i < pTy.mlir_tuple_type().types_size(); i++) {
      types.emplace_back(getType(mInfo, pTy.mlir_tuple_type().types(i)));
    }
    rTy = mlir::TupleType::get(ctx, types);
  } break;
  case MLIRType::TypeCase::kMlirUnrankedMemRefType:
    break;
  case MLIRType::TypeCase::kMlirUnrankedTensorType:
    break;
  case MLIRType::TypeCase::kCirExceptionType:
    rTy = cir::ExceptionInfoType::get(ctx);
    break;
  case MLIRType::TypeCase::kCirVoidType:
    rTy = cir::VoidType::get(ctx);
    break;
  case MLIRType::TypeCase::kCirStructType:
    llvm_unreachable("Definition of StructTypes should not happen"
                     "inside of a generic defineType!");
  case MLIRType::TypeCase::TYPE_NOT_SET:
    llvm_unreachable("Type kind not set!");
  default:
    llvm::errs() << "MLIRType::TypeCase set as " << pTy.type_case();
    llvm_unreachable("NYI");
  }
  auto typeId = pTy.id().id();
  mInfo.types[typeId] = rTy;
  llvm::errs() << "defined and saved " << "typeId" << "\n\n";
}

void Deserializer::defineIncompleteStruct(ModuleInfo &mInfo,
                                          const MLIRType &pTy) {
  assert(pTy.has_cir_struct_type() && "pTy is not of StructType!");
  auto nameAttr = AttrDeserializer::deserializeMLIRStringAttr(
      mInfo, pTy.cir_struct_type().name());
  auto recordKind =
      EnumDeserializer::deserializeCIRRecordKind(pTy.cir_struct_type().kind());
  auto incompleteStruct =
      cir::StructType::get(&mInfo.ctx, nameAttr, recordKind);
  mInfo.types[pTy.id().id()] = incompleteStruct;
}

void Deserializer::defineCompleteStruct(ModuleInfo &mInfo,
                                        const MLIRType &pTy) {
  assert(pTy.has_cir_struct_type() && "pTy is not of StructType");
  assert(!pTy.cir_struct_type().incomplete() && "incomplete struct received!");
  auto attrName = AttrDeserializer::deserializeMLIRStringAttr(
      mInfo, pTy.cir_struct_type().name());
  auto pRecordKind = pTy.cir_struct_type().kind();
  auto recordKind = EnumDeserializer::deserializeCIRRecordKind(pRecordKind);
  auto packed = pTy.cir_struct_type().packed();
  std::vector<mlir::Type> vecMemberTys;
  for (auto ty : pTy.cir_struct_type().members())
    vecMemberTys.push_back(getType(mInfo, ty));
  auto memberTys = mlir::ArrayRef<mlir::Type>(vecMemberTys);
  auto fullStruct =
      cir::StructType::get(&mInfo.ctx, memberTys, attrName, packed, recordKind);
  /* completion will fail if the data is mismatched with preexisting one */
  fullStruct.complete(memberTys, packed);
  mInfo.types[pTy.id().id()] = fullStruct;
}

void Deserializer::aggregateTypes(ModuleInfo &mInfo,
                                  const MLIRModule &pModule) {
  auto types = pModule.types();

  for (auto ty : types) {
    mInfo.serTypes[ty.id().id()] = ty;
    /* initiating incomplete structs beforehand to resolve recursive cases */
    if (ty.type_case() == MLIRType::TypeCase::kCirStructType) {
      defineIncompleteStruct(mInfo, ty);
    }
  }

  for (auto ty : types) {
    /* getType will define the type for us if it isn't yet */
    std::ignore = getType(mInfo, ty.id());
  }

  for (auto ty : types) {
    /* completing the definition of structs */
    if (ty.type_case() == MLIRType::TypeCase::kCirStructType) {
      defineCompleteStruct(mInfo, ty);
    }
  }
}

void Deserializer::deserializeBlock(FunctionInfo &fInfo,
                                    const MLIRBlock &pBlock,
                                    bool isEntryBlock) {
  auto *bb = fInfo.blocks[pBlock.id().id()];
  fInfo.owner.builder.setInsertionPointToStart(bb);
  for (const auto &pOp : pBlock.operations()) {
    mlir::Operation *op;
    if (pOp.has_get_global_op()) {
      auto symName = pOp.get_global_op().name().root_reference().value();
      assert(fInfo.owner.globals.count(symName));
      op = fInfo.owner.builder.createGetGlobal(mlir::dyn_cast<cir::GlobalOp>(fInfo.owner.globals.at(symName))).getDefiningOp();
    }
    else {
      op = OpDeserializer::deserializeMLIROp(fInfo, fInfo.owner, pOp);

      op->setLoc(
          AttrDeserializer::deserializeMLIRLocation(fInfo.owner, pOp.location()));
      llvm::errs() << "loc: ";
      op->getLoc().print(llvm::errs());
      llvm::errs() << "\n\n";
    }
    fInfo.ops[pOp.id().id()] = op;
  }
  // entry blocks contain function's arguments upon their construction
  if (isEntryBlock) {
    assert(bb->getArguments().size() == pBlock.argument_types_size());
    return;
  }
  for (const auto &arg : pBlock.argument_types()) {
    llvm::errs() << "arg present: " << arg.id() << "\n\n";
    auto argType = getType(fInfo.owner, arg);
    bb->addArgument(argType, fInfo.owner.builder.getUnknownLoc());
  }
}

void Deserializer::deserializeFunc(ModuleInfo &mInfo,
                                   const CIRFunction &pFunc) {
  auto funcInfo = FunctionInfo(mInfo);
  auto funcOp =
      OpDeserializer::deserializeCIRFuncOp(funcInfo, mInfo, pFunc.info());
  if (pFunc.blocks().block_size() > 0) {
    auto bb = pFunc.blocks().block(0);
    funcInfo.blocks[bb.id().id()] = funcOp.addEntryBlock();
    deserializeBlock(funcInfo, bb, /*isEntryBlock=*/true);
  }
  for (int bbId = 1; bbId < pFunc.blocks().block_size(); ++bbId) {
    auto bb = pFunc.blocks().block(bbId);
    funcInfo.blocks[bb.id().id()] = funcOp.addBlock();
    deserializeBlock(funcInfo, bb, /*isEntryBlock=*/false);
  }
  mInfo.module.push_back(funcOp);
  llvm::errs() << "entry block size: " << funcOp.getBlocks().front().getArguments().size() << "\n\n";
  mInfo.funcs[pFunc.id().id()] = &funcOp;
}

mlir::Block *Deserializer::getBlock(FunctionInfo &fInfo,
                                    const MLIRBlockID &pBlock) {
  auto blockId = pBlock.id();
  assert(fInfo.blocks.count(blockId) &&
         "blockId is not present in Block cache!");
  return fInfo.blocks.at(blockId);
}

mlir::Value Deserializer::deserializeValue(FunctionInfo &fInfo,
                                           const MLIRValue &pValue) {
  switch (pValue.value_case()) {
  case MLIRValue::kOpResult: {
    auto opId = pValue.op_result().owner().id();
    auto resultId = pValue.op_result().result_number();
    assert(fInfo.ops.count(opId) && "opId is not present in Operation cache!");
    return fInfo.ops.at(opId)->getOpResult(resultId);
  } break;
  case MLIRValue::kBlockArgument: {
    auto blockId = pValue.block_argument().owner().id();
    auto argId = pValue.block_argument().arg_number();
    assert(fInfo.blocks.count(blockId) &&
           "blockId is not present in Block cache!");
    return fInfo.blocks.at(blockId)->getArgument(argId);
  } break;
  case MLIRValue::VALUE_NOT_SET:
    llvm_unreachable("Unexpected value case for MLIRValue!");
    break;
  }
}

void Deserializer::deserializeGlobal(ModuleInfo &mInfo,
                                     const CIRGlobal &pGlobal) {
  // global ops exist outside of FuncOps
  // however, OpDeserializer expects it as an argument
  // creating an empty instance to keep uniformity
  FunctionInfo emptyFuncInfo(mInfo);
  auto op = OpDeserializer::deserializeCIRGlobalOp(emptyFuncInfo, mInfo,
                                                   pGlobal.info());
  mInfo.globals[pGlobal.id().id()] = op.getOperation();

  llvm::errs() << "global sym name: " << op.getName() << "\n\n";
}

mlir::ModuleOp Deserializer::deserializeModule(mlir::MLIRContext &ctx,
                                               const MLIRModule &pModule) {
  auto builder = cir::CIRBaseBuilderTy(ctx);
  auto newModule =
      mlir::ModuleOp::create(builder.getUnknownLoc(), pModule.id().id());
  cir::CIRDataLayout dataLayout(newModule);

  auto mInfo = ModuleInfo(ctx, builder, dataLayout, newModule);

  llvm::errs() << "aggregating types\n\n";
  aggregateTypes(mInfo, pModule);

  llvm::errs() << "deserializing globals\n\n";
  for (const auto &pGlobal : pModule.globals()) {
    deserializeGlobal(mInfo, pGlobal);
  }
  llvm::errs() << "setting attributes\n\n";
  for (const auto &pAttr : pModule.attributes()) {
    auto namedAttr = AttrDeserializer::deserializeMLIRNamedAttr(mInfo, pAttr);
    newModule->setAttr(namedAttr.getName(), namedAttr.getValue());
  }
  llvm::errs() << "deserializing funcs\n\n";
  for (const auto &pFunc : pModule.functions()) {
    deserializeFunc(mInfo, pFunc);
  }

  llvm::errs() << "verifying the final result\n\n";
  assert(mlir::verify(newModule).succeeded());

  return newModule;
}
