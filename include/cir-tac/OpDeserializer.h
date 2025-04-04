/* Autogenerated by mlir-tblgen; don't manually edit. */
// clang-format off

#pragma once

#include "Util.h"
#include "proto/op.pb.h"
#include "proto/setup.pb.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/IR/Block.h>

namespace protocir {

class OpDeserializer {
public:
  static mlir::Operation *deserializeMLIROp(FunctionInfo &fInfo, ModuleInfo &mInfo, MLIROp pOp);
  static cir::AbsOp deserializeCIRAbsOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAbsOp pOp);
  static cir::AllocExceptionOp deserializeCIRAllocExceptionOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAllocExceptionOp pOp);
  static cir::AllocaOp deserializeCIRAllocaOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAllocaOp pOp);
  static cir::ArrayCtor deserializeCIRArrayCtorOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRArrayCtorOp pOp);
  static cir::ArrayDtor deserializeCIRArrayDtorOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRArrayDtorOp pOp);
  static cir::AssumeAlignedOp deserializeCIRAssumeAlignedOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAssumeAlignedOp pOp);
  static cir::AssumeOp deserializeCIRAssumeOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAssumeOp pOp);
  static cir::AssumeSepStorageOp deserializeCIRAssumeSepStorageOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAssumeSepStorageOp pOp);
  static cir::AtomicCmpXchg deserializeCIRAtomicCmpXchgOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAtomicCmpXchgOp pOp);
  static cir::AtomicFetch deserializeCIRAtomicFetchOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAtomicFetchOp pOp);
  static cir::AtomicXchg deserializeCIRAtomicXchgOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAtomicXchgOp pOp);
  static cir::AwaitOp deserializeCIRAwaitOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRAwaitOp pOp);
  static cir::BaseClassAddrOp deserializeCIRBaseClassAddrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBaseClassAddrOp pOp);
  static cir::BinOp deserializeCIRBinOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBinOp pOp);
  static cir::BinOpOverflowOp deserializeCIRBinOpOverflowOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBinOpOverflowOp pOp);
  static cir::BitClrsbOp deserializeCIRBitClrsbOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitClrsbOp pOp);
  static cir::BitClzOp deserializeCIRBitClzOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitClzOp pOp);
  static cir::BitCtzOp deserializeCIRBitCtzOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitCtzOp pOp);
  static cir::BitFfsOp deserializeCIRBitFfsOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitFfsOp pOp);
  static cir::BitParityOp deserializeCIRBitParityOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitParityOp pOp);
  static cir::BitPopcountOp deserializeCIRBitPopcountOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBitPopcountOp pOp);
  static cir::BrCondOp deserializeCIRBrCondOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBrCondOp pOp);
  static cir::BrOp deserializeCIRBrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBrOp pOp);
  static cir::BreakOp deserializeCIRBreakOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRBreakOp pOp);
  static cir::ByteswapOp deserializeCIRByteswapOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRByteswapOp pOp);
  static cir::InlineAsmOp deserializeCIRInlineAsmOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRInlineAsmOp pOp);
  static cir::CallOp deserializeCIRCallOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCallOp pOp);
  static cir::CaseOp deserializeCIRCaseOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCaseOp pOp);
  static cir::CastOp deserializeCIRCastOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCastOp pOp);
  static cir::CatchParamOp deserializeCIRCatchParamOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCatchParamOp pOp);
  static cir::CeilOp deserializeCIRCeilOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCeilOp pOp);
  static cir::ClearCacheOp deserializeCIRClearCacheOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRClearCacheOp pOp);
  static cir::CmpOp deserializeCIRCmpOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCmpOp pOp);
  static cir::CmpThreeWayOp deserializeCIRCmpThreeWayOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCmpThreeWayOp pOp);
  static cir::ComplexBinOp deserializeCIRComplexBinOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexBinOp pOp);
  static cir::ComplexCreateOp deserializeCIRComplexCreateOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexCreateOp pOp);
  static cir::ComplexImagOp deserializeCIRComplexImagOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexImagOp pOp);
  static cir::ComplexImagPtrOp deserializeCIRComplexImagPtrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexImagPtrOp pOp);
  static cir::ComplexRealOp deserializeCIRComplexRealOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexRealOp pOp);
  static cir::ComplexRealPtrOp deserializeCIRComplexRealPtrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRComplexRealPtrOp pOp);
  static cir::ConditionOp deserializeCIRConditionOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRConditionOp pOp);
  static cir::ConstantOp deserializeCIRConstantOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRConstantOp pOp);
  static cir::ContinueOp deserializeCIRContinueOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRContinueOp pOp);
  static cir::CopyOp deserializeCIRCopyOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCopyOp pOp);
  static cir::CopysignOp deserializeCIRCopysignOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCopysignOp pOp);
  static cir::CosOp deserializeCIRCosOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRCosOp pOp);
  static cir::DerivedClassAddrOp deserializeCIRDerivedClassAddrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRDerivedClassAddrOp pOp);
  static cir::DoWhileOp deserializeCIRDoWhileOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRDoWhileOp pOp);
  static cir::DynamicCastOp deserializeCIRDynamicCastOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRDynamicCastOp pOp);
  static cir::EhInflightOp deserializeCIREhInflightOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIREhInflightOp pOp);
  static cir::EhTypeIdOp deserializeCIREhTypeIdOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIREhTypeIdOp pOp);
  static cir::Exp2Op deserializeCIRExp2Op(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRExp2Op pOp);
  static cir::ExpOp deserializeCIRExpOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRExpOp pOp);
  static cir::ExpectOp deserializeCIRExpectOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRExpectOp pOp);
  static cir::FAbsOp deserializeCIRFAbsOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFAbsOp pOp);
  static cir::FMaxOp deserializeCIRFMaxOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFMaxOp pOp);
  static cir::FMinOp deserializeCIRFMinOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFMinOp pOp);
  static cir::FModOp deserializeCIRFModOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFModOp pOp);
  static cir::FloorOp deserializeCIRFloorOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFloorOp pOp);
  static cir::ForOp deserializeCIRForOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRForOp pOp);
  static cir::FrameAddrOp deserializeCIRFrameAddrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFrameAddrOp pOp);
  static cir::FreeExceptionOp deserializeCIRFreeExceptionOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFreeExceptionOp pOp);
  static cir::FuncOp deserializeCIRFuncOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRFuncOp pOp);
  static cir::GetBitfieldOp deserializeCIRGetBitfieldOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGetBitfieldOp pOp);
  static cir::GetGlobalOp deserializeCIRGetGlobalOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGetGlobalOp pOp);
  static cir::GetMemberOp deserializeCIRGetMemberOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGetMemberOp pOp);
  static cir::GetMethodOp deserializeCIRGetMethodOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGetMethodOp pOp);
  static cir::GetRuntimeMemberOp deserializeCIRGetRuntimeMemberOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGetRuntimeMemberOp pOp);
  static cir::GlobalOp deserializeCIRGlobalOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGlobalOp pOp);
  static cir::GotoOp deserializeCIRGotoOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRGotoOp pOp);
  static cir::IfOp deserializeCIRIfOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRIfOp pOp);
  static cir::IsConstantOp deserializeCIRIsConstantOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRIsConstantOp pOp);
  static cir::IsFPClassOp deserializeCIRIsFPClassOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRIsFPClassOp pOp);
  static cir::IterBeginOp deserializeCIRIterBeginOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRIterBeginOp pOp);
  static cir::IterEndOp deserializeCIRIterEndOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRIterEndOp pOp);
  static cir::LLVMIntrinsicCallOp deserializeCIRLLVMIntrinsicCallOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLLVMIntrinsicCallOp pOp);
  static cir::LLrintOp deserializeCIRLLrintOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLLrintOp pOp);
  static cir::LLroundOp deserializeCIRLLroundOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLLroundOp pOp);
  static cir::LabelOp deserializeCIRLabelOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLabelOp pOp);
  static cir::LoadOp deserializeCIRLoadOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLoadOp pOp);
  static cir::Log10Op deserializeCIRLog10Op(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLog10Op pOp);
  static cir::Log2Op deserializeCIRLog2Op(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLog2Op pOp);
  static cir::LogOp deserializeCIRLogOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLogOp pOp);
  static cir::LrintOp deserializeCIRLrintOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLrintOp pOp);
  static cir::LroundOp deserializeCIRLroundOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRLroundOp pOp);
  static cir::MemChrOp deserializeCIRMemChrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemChrOp pOp);
  static cir::MemCpyInlineOp deserializeCIRMemCpyInlineOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemCpyInlineOp pOp);
  static cir::MemCpyOp deserializeCIRMemCpyOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemCpyOp pOp);
  static cir::MemMoveOp deserializeCIRMemMoveOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemMoveOp pOp);
  static cir::MemSetInlineOp deserializeCIRMemSetInlineOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemSetInlineOp pOp);
  static cir::MemSetOp deserializeCIRMemSetOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRMemSetOp pOp);
  static cir::NearbyintOp deserializeCIRNearbyintOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRNearbyintOp pOp);
  static cir::ObjSizeOp deserializeCIRObjSizeOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRObjSizeOp pOp);
  static cir::PowOp deserializeCIRPowOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRPowOp pOp);
  static cir::PrefetchOp deserializeCIRPrefetchOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRPrefetchOp pOp);
  static cir::PtrDiffOp deserializeCIRPtrDiffOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRPtrDiffOp pOp);
  static cir::PtrMaskOp deserializeCIRPtrMaskOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRPtrMaskOp pOp);
  static cir::PtrStrideOp deserializeCIRPtrStrideOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRPtrStrideOp pOp);
  static cir::ResumeOp deserializeCIRResumeOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRResumeOp pOp);
  static cir::ReturnAddrOp deserializeCIRReturnAddrOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRReturnAddrOp pOp);
  static cir::ReturnOp deserializeCIRReturnOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRReturnOp pOp);
  static cir::RintOp deserializeCIRRintOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRRintOp pOp);
  static cir::RotateOp deserializeCIRRotateOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRRotateOp pOp);
  static cir::RoundOp deserializeCIRRoundOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRRoundOp pOp);
  static cir::ScopeOp deserializeCIRScopeOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRScopeOp pOp);
  static cir::SelectOp deserializeCIRSelectOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSelectOp pOp);
  static cir::SetBitfieldOp deserializeCIRSetBitfieldOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSetBitfieldOp pOp);
  static cir::ShiftOp deserializeCIRShiftOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRShiftOp pOp);
  static cir::SignBitOp deserializeCIRSignBitOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSignBitOp pOp);
  static cir::SinOp deserializeCIRSinOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSinOp pOp);
  static cir::SqrtOp deserializeCIRSqrtOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSqrtOp pOp);
  static cir::StackRestoreOp deserializeCIRStackRestoreOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRStackRestoreOp pOp);
  static cir::StackSaveOp deserializeCIRStackSaveOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRStackSaveOp pOp);
  static cir::StdFindOp deserializeCIRStdFindOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRStdFindOp pOp);
  static cir::StdInitializerListOp deserializeCIRStdInitializerListOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRStdInitializerListOp pOp);
  static cir::StoreOp deserializeCIRStoreOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRStoreOp pOp);
  static cir::SwitchFlatOp deserializeCIRSwitchFlatOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSwitchFlatOp pOp);
  static cir::SwitchOp deserializeCIRSwitchOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRSwitchOp pOp);
  static cir::TernaryOp deserializeCIRTernaryOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRTernaryOp pOp);
  static cir::ThrowOp deserializeCIRThrowOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRThrowOp pOp);
  static cir::TrapOp deserializeCIRTrapOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRTrapOp pOp);
  static cir::TruncOp deserializeCIRTruncOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRTruncOp pOp);
  static cir::TryCallOp deserializeCIRTryCallOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRTryCallOp pOp);
  static cir::TryOp deserializeCIRTryOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRTryOp pOp);
  static cir::UnaryOp deserializeCIRUnaryOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRUnaryOp pOp);
  static cir::UnreachableOp deserializeCIRUnreachableOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRUnreachableOp pOp);
  static cir::VAArgOp deserializeCIRVAArgOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVAArgOp pOp);
  static cir::VACopyOp deserializeCIRVACopyOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVACopyOp pOp);
  static cir::VAEndOp deserializeCIRVAEndOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVAEndOp pOp);
  static cir::VAStartOp deserializeCIRVAStartOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVAStartOp pOp);
  static cir::VTTAddrPointOp deserializeCIRVTTAddrPointOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVTTAddrPointOp pOp);
  static cir::VTableAddrPointOp deserializeCIRVTableAddrPointOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVTableAddrPointOp pOp);
  static cir::VecCmpOp deserializeCIRVecCmpOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecCmpOp pOp);
  static cir::VecCreateOp deserializeCIRVecCreateOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecCreateOp pOp);
  static cir::VecExtractOp deserializeCIRVecExtractOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecExtractOp pOp);
  static cir::VecInsertOp deserializeCIRVecInsertOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecInsertOp pOp);
  static cir::VecShuffleDynamicOp deserializeCIRVecShuffleDynamicOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecShuffleDynamicOp pOp);
  static cir::VecShuffleOp deserializeCIRVecShuffleOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecShuffleOp pOp);
  static cir::VecSplatOp deserializeCIRVecSplatOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecSplatOp pOp);
  static cir::VecTernaryOp deserializeCIRVecTernaryOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRVecTernaryOp pOp);
  static cir::WhileOp deserializeCIRWhileOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRWhileOp pOp);
  static cir::YieldOp deserializeCIRYieldOp(FunctionInfo &fInfo, ModuleInfo &mInfo, CIRYieldOp pOp);
};

} // namespace protocir
// clang-format on
