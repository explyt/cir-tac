/* Autogenerated by mlir-tblgen; don't manually edit. */

#include "cir-tac/Serializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;

void Serializer::serializeOperation(mlir::Operation &inst,
                                    protocir::CIROp *pInst,
                                    protocir::CIRModuleID pModuleID,
                                    TypeCache &typeCache,
                                    OperationCache &opCache,
                                    FunctionCache &functionCache

) {
  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)

      .Case<cir::AbsOp>([instID, pInst, pModuleID, &typeCache](cir::AbsOp op) {
        protocir::CIRAbsOp pAbsOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_abs_op()->CopyFrom(pAbsOp);
      })

      .Case<cir::AllocExceptionOp>([instID, pInst, pModuleID, &typeCache](cir::AllocExceptionOp op) {
        protocir::CIRAllocExceptionOp pAllocExceptionOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_alloc_exception_op()->CopyFrom(pAllocExceptionOp);
      })

      .Case<cir::AllocaOp>([instID, pInst, pModuleID, &typeCache](cir::AllocaOp op) {
        protocir::CIRAllocaOp pAllocaOp;
        pInst->mutable_base()->set_id(instID);

        auto allocaType = op.getAllocaType();
        protocir::CIRTypeID pAllocaTypeID;
        *pAllocaTypeID.mutable_module_id() = pModuleID;
        pAllocaTypeID.set_id(internType(typeCache, allocaType));
        *pAllocaOp.mutable_allocatype() = pAllocaTypeID;

        auto name = op.getName();
        *pAllocaOp.mutable_name() = name.str();

        auto init = op.getInit();
        pAllocaOp.set_init(init);

        auto constant = op.getConstant();
        pAllocaOp.set_constant(constant);

        auto alignment = op.getAlignment();
        if (alignment) {
          pAllocaOp.set_alignment(alignment.value());
        }

        pInst->mutable_alloca_op()->CopyFrom(pAllocaOp);
      })

      .Case<cir::ArrayCtor>([instID, pInst, pModuleID, &typeCache](cir::ArrayCtor op) {
        protocir::CIRArrayCtor pArrayCtor;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_array_ctor()->CopyFrom(pArrayCtor);
      })

      .Case<cir::ArrayDtor>([instID, pInst, pModuleID, &typeCache](cir::ArrayDtor op) {
        protocir::CIRArrayDtor pArrayDtor;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_array_dtor()->CopyFrom(pArrayDtor);
      })

      .Case<cir::AssumeAlignedOp>([instID, pInst, pModuleID, &typeCache](cir::AssumeAlignedOp op) {
        protocir::CIRAssumeAlignedOp pAssumeAlignedOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_assume_aligned_op()->CopyFrom(pAssumeAlignedOp);
      })

      .Case<cir::AssumeOp>([instID, pInst, pModuleID, &typeCache](cir::AssumeOp op) {
        protocir::CIRAssumeOp pAssumeOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_assume_op()->CopyFrom(pAssumeOp);
      })

      .Case<cir::AssumeSepStorageOp>([instID, pInst, pModuleID, &typeCache](cir::AssumeSepStorageOp op) {
        protocir::CIRAssumeSepStorageOp pAssumeSepStorageOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_assume_sep_storage_op()->CopyFrom(pAssumeSepStorageOp);
      })

      .Case<cir::AtomicCmpXchg>([instID, pInst, pModuleID, &typeCache](cir::AtomicCmpXchg op) {
        protocir::CIRAtomicCmpXchg pAtomicCmpXchg;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_atomic_cmp_xchg()->CopyFrom(pAtomicCmpXchg);
      })

      .Case<cir::AtomicFetch>([instID, pInst, pModuleID, &typeCache](cir::AtomicFetch op) {
        protocir::CIRAtomicFetch pAtomicFetch;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_atomic_fetch()->CopyFrom(pAtomicFetch);
      })

      .Case<cir::AtomicXchg>([instID, pInst, pModuleID, &typeCache](cir::AtomicXchg op) {
        protocir::CIRAtomicXchg pAtomicXchg;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_atomic_xchg()->CopyFrom(pAtomicXchg);
      })

      .Case<cir::AwaitOp>([instID, pInst, pModuleID, &typeCache](cir::AwaitOp op) {
        protocir::CIRAwaitOp pAwaitOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_await_op()->CopyFrom(pAwaitOp);
      })

      .Case<cir::BaseClassAddrOp>([instID, pInst, pModuleID, &typeCache](cir::BaseClassAddrOp op) {
        protocir::CIRBaseClassAddrOp pBaseClassAddrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_base_class_addr_op()->CopyFrom(pBaseClassAddrOp);
      })

      .Case<cir::BinOp>([instID, pInst, pModuleID, &typeCache](cir::BinOp op) {
        protocir::CIRBinOp pBinOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bin_op()->CopyFrom(pBinOp);
      })

      .Case<cir::BinOpOverflowOp>([instID, pInst, pModuleID, &typeCache](cir::BinOpOverflowOp op) {
        protocir::CIRBinOpOverflowOp pBinOpOverflowOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bin_op_overflow_op()->CopyFrom(pBinOpOverflowOp);
      })

      .Case<cir::BitClrsbOp>([instID, pInst, pModuleID, &typeCache](cir::BitClrsbOp op) {
        protocir::CIRBitClrsbOp pBitClrsbOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_clrsb_op()->CopyFrom(pBitClrsbOp);
      })

      .Case<cir::BitClzOp>([instID, pInst, pModuleID, &typeCache](cir::BitClzOp op) {
        protocir::CIRBitClzOp pBitClzOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_clz_op()->CopyFrom(pBitClzOp);
      })

      .Case<cir::BitCtzOp>([instID, pInst, pModuleID, &typeCache](cir::BitCtzOp op) {
        protocir::CIRBitCtzOp pBitCtzOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_ctz_op()->CopyFrom(pBitCtzOp);
      })

      .Case<cir::BitFfsOp>([instID, pInst, pModuleID, &typeCache](cir::BitFfsOp op) {
        protocir::CIRBitFfsOp pBitFfsOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_ffs_op()->CopyFrom(pBitFfsOp);
      })

      .Case<cir::BitParityOp>([instID, pInst, pModuleID, &typeCache](cir::BitParityOp op) {
        protocir::CIRBitParityOp pBitParityOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_parity_op()->CopyFrom(pBitParityOp);
      })

      .Case<cir::BitPopcountOp>([instID, pInst, pModuleID, &typeCache](cir::BitPopcountOp op) {
        protocir::CIRBitPopcountOp pBitPopcountOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_bit_popcount_op()->CopyFrom(pBitPopcountOp);
      })

      .Case<cir::BrCondOp>([instID, pInst, pModuleID, &typeCache](cir::BrCondOp op) {
        protocir::CIRBrCondOp pBrCondOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_br_cond_op()->CopyFrom(pBrCondOp);
      })

      .Case<cir::BrOp>([instID, pInst, pModuleID, &typeCache](cir::BrOp op) {
        protocir::CIRBrOp pBrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_br_op()->CopyFrom(pBrOp);
      })

      .Case<cir::BreakOp>([instID, pInst, pModuleID, &typeCache](cir::BreakOp op) {
        protocir::CIRBreakOp pBreakOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_break_op()->CopyFrom(pBreakOp);
      })

      .Case<cir::ByteswapOp>([instID, pInst, pModuleID, &typeCache](cir::ByteswapOp op) {
        protocir::CIRByteswapOp pByteswapOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_byteswap_op()->CopyFrom(pByteswapOp);
      })

      .Case<cir::InlineAsmOp>([instID, pInst, pModuleID, &typeCache](cir::InlineAsmOp op) {
        protocir::CIRInlineAsmOp pInlineAsmOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_inline_asm_op()->CopyFrom(pInlineAsmOp);
      })

      .Case<cir::CallOp>([instID, pInst, pModuleID, &typeCache](cir::CallOp op) {
        protocir::CIRCallOp pCallOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_call_op()->CopyFrom(pCallOp);
      })

      .Case<cir::CaseOp>([instID, pInst, pModuleID, &typeCache](cir::CaseOp op) {
        protocir::CIRCaseOp pCaseOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_case_op()->CopyFrom(pCaseOp);
      })

      .Case<cir::CastOp>([instID, pInst, pModuleID, &typeCache](cir::CastOp op) {
        protocir::CIRCastOp pCastOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_cast_op()->CopyFrom(pCastOp);
      })

      .Case<cir::CatchParamOp>([instID, pInst, pModuleID, &typeCache](cir::CatchParamOp op) {
        protocir::CIRCatchParamOp pCatchParamOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_catch_param_op()->CopyFrom(pCatchParamOp);
      })

      .Case<cir::CeilOp>([instID, pInst, pModuleID, &typeCache](cir::CeilOp op) {
        protocir::CIRCeilOp pCeilOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_ceil_op()->CopyFrom(pCeilOp);
      })

      .Case<cir::ClearCacheOp>([instID, pInst, pModuleID, &typeCache](cir::ClearCacheOp op) {
        protocir::CIRClearCacheOp pClearCacheOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_clear_cache_op()->CopyFrom(pClearCacheOp);
      })

      .Case<cir::CmpOp>([instID, pInst, pModuleID, &typeCache](cir::CmpOp op) {
        protocir::CIRCmpOp pCmpOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_cmp_op()->CopyFrom(pCmpOp);
      })

      .Case<cir::CmpThreeWayOp>([instID, pInst, pModuleID, &typeCache](cir::CmpThreeWayOp op) {
        protocir::CIRCmpThreeWayOp pCmpThreeWayOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_cmp_three_way_op()->CopyFrom(pCmpThreeWayOp);
      })

      .Case<cir::ComplexBinOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexBinOp op) {
        protocir::CIRComplexBinOp pComplexBinOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_bin_op()->CopyFrom(pComplexBinOp);
      })

      .Case<cir::ComplexCreateOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexCreateOp op) {
        protocir::CIRComplexCreateOp pComplexCreateOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_create_op()->CopyFrom(pComplexCreateOp);
      })

      .Case<cir::ComplexImagOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexImagOp op) {
        protocir::CIRComplexImagOp pComplexImagOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_imag_op()->CopyFrom(pComplexImagOp);
      })

      .Case<cir::ComplexImagPtrOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexImagPtrOp op) {
        protocir::CIRComplexImagPtrOp pComplexImagPtrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_imag_ptr_op()->CopyFrom(pComplexImagPtrOp);
      })

      .Case<cir::ComplexRealOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexRealOp op) {
        protocir::CIRComplexRealOp pComplexRealOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_real_op()->CopyFrom(pComplexRealOp);
      })

      .Case<cir::ComplexRealPtrOp>([instID, pInst, pModuleID, &typeCache](cir::ComplexRealPtrOp op) {
        protocir::CIRComplexRealPtrOp pComplexRealPtrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_complex_real_ptr_op()->CopyFrom(pComplexRealPtrOp);
      })

      .Case<cir::ConditionOp>([instID, pInst, pModuleID, &typeCache](cir::ConditionOp op) {
        protocir::CIRConditionOp pConditionOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_condition_op()->CopyFrom(pConditionOp);
      })

      .Case<cir::ConstantOp>([instID, pInst, pModuleID, &typeCache](cir::ConstantOp op) {
        protocir::CIRConstantOp pConstantOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_constant_op()->CopyFrom(pConstantOp);
      })

      .Case<cir::ContinueOp>([instID, pInst, pModuleID, &typeCache](cir::ContinueOp op) {
        protocir::CIRContinueOp pContinueOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_continue_op()->CopyFrom(pContinueOp);
      })

      .Case<cir::CopyOp>([instID, pInst, pModuleID, &typeCache](cir::CopyOp op) {
        protocir::CIRCopyOp pCopyOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_copy_op()->CopyFrom(pCopyOp);
      })

      .Case<cir::CopysignOp>([instID, pInst, pModuleID, &typeCache](cir::CopysignOp op) {
        protocir::CIRCopysignOp pCopysignOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_copysign_op()->CopyFrom(pCopysignOp);
      })

      .Case<cir::CosOp>([instID, pInst, pModuleID, &typeCache](cir::CosOp op) {
        protocir::CIRCosOp pCosOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_cos_op()->CopyFrom(pCosOp);
      })

      .Case<cir::DerivedClassAddrOp>([instID, pInst, pModuleID, &typeCache](cir::DerivedClassAddrOp op) {
        protocir::CIRDerivedClassAddrOp pDerivedClassAddrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_derived_class_addr_op()->CopyFrom(pDerivedClassAddrOp);
      })

      .Case<cir::DoWhileOp>([instID, pInst, pModuleID, &typeCache](cir::DoWhileOp op) {
        protocir::CIRDoWhileOp pDoWhileOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_do_while_op()->CopyFrom(pDoWhileOp);
      })

      .Case<cir::DynamicCastOp>([instID, pInst, pModuleID, &typeCache](cir::DynamicCastOp op) {
        protocir::CIRDynamicCastOp pDynamicCastOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_dynamic_cast_op()->CopyFrom(pDynamicCastOp);
      })

      .Case<cir::EhInflightOp>([instID, pInst, pModuleID, &typeCache](cir::EhInflightOp op) {
        protocir::CIREhInflightOp pEhInflightOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_eh_inflight_op()->CopyFrom(pEhInflightOp);
      })

      .Case<cir::EhTypeIdOp>([instID, pInst, pModuleID, &typeCache](cir::EhTypeIdOp op) {
        protocir::CIREhTypeIdOp pEhTypeIdOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_eh_type_id_op()->CopyFrom(pEhTypeIdOp);
      })

      .Case<cir::Exp2Op>([instID, pInst, pModuleID, &typeCache](cir::Exp2Op op) {
        protocir::CIRExp2Op pExp2Op;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_exp2_op()->CopyFrom(pExp2Op);
      })

      .Case<cir::ExpOp>([instID, pInst, pModuleID, &typeCache](cir::ExpOp op) {
        protocir::CIRExpOp pExpOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_exp_op()->CopyFrom(pExpOp);
      })

      .Case<cir::ExpectOp>([instID, pInst, pModuleID, &typeCache](cir::ExpectOp op) {
        protocir::CIRExpectOp pExpectOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_expect_op()->CopyFrom(pExpectOp);
      })

      .Case<cir::FAbsOp>([instID, pInst, pModuleID, &typeCache](cir::FAbsOp op) {
        protocir::CIRFAbsOp pFAbsOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_f_abs_op()->CopyFrom(pFAbsOp);
      })

      .Case<cir::FMaxOp>([instID, pInst, pModuleID, &typeCache](cir::FMaxOp op) {
        protocir::CIRFMaxOp pFMaxOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_f_max_op()->CopyFrom(pFMaxOp);
      })

      .Case<cir::FMinOp>([instID, pInst, pModuleID, &typeCache](cir::FMinOp op) {
        protocir::CIRFMinOp pFMinOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_f_min_op()->CopyFrom(pFMinOp);
      })

      .Case<cir::FModOp>([instID, pInst, pModuleID, &typeCache](cir::FModOp op) {
        protocir::CIRFModOp pFModOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_f_mod_op()->CopyFrom(pFModOp);
      })

      .Case<cir::FloorOp>([instID, pInst, pModuleID, &typeCache](cir::FloorOp op) {
        protocir::CIRFloorOp pFloorOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_floor_op()->CopyFrom(pFloorOp);
      })

      .Case<cir::ForOp>([instID, pInst, pModuleID, &typeCache](cir::ForOp op) {
        protocir::CIRForOp pForOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_for_op()->CopyFrom(pForOp);
      })

      .Case<cir::FrameAddrOp>([instID, pInst, pModuleID, &typeCache](cir::FrameAddrOp op) {
        protocir::CIRFrameAddrOp pFrameAddrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_frame_addr_op()->CopyFrom(pFrameAddrOp);
      })

      .Case<cir::FreeExceptionOp>([instID, pInst, pModuleID, &typeCache](cir::FreeExceptionOp op) {
        protocir::CIRFreeExceptionOp pFreeExceptionOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_free_exception_op()->CopyFrom(pFreeExceptionOp);
      })

      .Case<cir::FuncOp>([instID, pInst, pModuleID, &typeCache](cir::FuncOp op) {
        protocir::CIRFuncOp pFuncOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_func_op()->CopyFrom(pFuncOp);
      })

      .Case<cir::GetBitfieldOp>([instID, pInst, pModuleID, &typeCache](cir::GetBitfieldOp op) {
        protocir::CIRGetBitfieldOp pGetBitfieldOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_get_bitfield_op()->CopyFrom(pGetBitfieldOp);
      })

      .Case<cir::GetGlobalOp>([instID, pInst, pModuleID, &typeCache](cir::GetGlobalOp op) {
        protocir::CIRGetGlobalOp pGetGlobalOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_get_global_op()->CopyFrom(pGetGlobalOp);
      })

      .Case<cir::GetMemberOp>([instID, pInst, pModuleID, &typeCache](cir::GetMemberOp op) {
        protocir::CIRGetMemberOp pGetMemberOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_get_member_op()->CopyFrom(pGetMemberOp);
      })

      .Case<cir::GetMethodOp>([instID, pInst, pModuleID, &typeCache](cir::GetMethodOp op) {
        protocir::CIRGetMethodOp pGetMethodOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_get_method_op()->CopyFrom(pGetMethodOp);
      })

      .Case<cir::GetRuntimeMemberOp>([instID, pInst, pModuleID, &typeCache](cir::GetRuntimeMemberOp op) {
        protocir::CIRGetRuntimeMemberOp pGetRuntimeMemberOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_get_runtime_member_op()->CopyFrom(pGetRuntimeMemberOp);
      })

      .Case<cir::GlobalOp>([instID, pInst, pModuleID, &typeCache](cir::GlobalOp op) {
        protocir::CIRGlobalOp pGlobalOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_global_op()->CopyFrom(pGlobalOp);
      })

      .Case<cir::GotoOp>([instID, pInst, pModuleID, &typeCache](cir::GotoOp op) {
        protocir::CIRGotoOp pGotoOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_goto_op()->CopyFrom(pGotoOp);
      })

      .Case<cir::IfOp>([instID, pInst, pModuleID, &typeCache](cir::IfOp op) {
        protocir::CIRIfOp pIfOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_if_op()->CopyFrom(pIfOp);
      })

      .Case<cir::IsConstantOp>([instID, pInst, pModuleID, &typeCache](cir::IsConstantOp op) {
        protocir::CIRIsConstantOp pIsConstantOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_is_constant_op()->CopyFrom(pIsConstantOp);
      })

      .Case<cir::IsFPClassOp>([instID, pInst, pModuleID, &typeCache](cir::IsFPClassOp op) {
        protocir::CIRIsFPClassOp pIsFPClassOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_is_fp_class_op()->CopyFrom(pIsFPClassOp);
      })

      .Case<cir::IterBeginOp>([instID, pInst, pModuleID, &typeCache](cir::IterBeginOp op) {
        protocir::CIRIterBeginOp pIterBeginOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_iter_begin_op()->CopyFrom(pIterBeginOp);
      })

      .Case<cir::IterEndOp>([instID, pInst, pModuleID, &typeCache](cir::IterEndOp op) {
        protocir::CIRIterEndOp pIterEndOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_iter_end_op()->CopyFrom(pIterEndOp);
      })

      .Case<cir::LLVMIntrinsicCallOp>([instID, pInst, pModuleID, &typeCache](cir::LLVMIntrinsicCallOp op) {
        protocir::CIRLLVMIntrinsicCallOp pLLVMIntrinsicCallOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_llvm_intrinsic_call_op()->CopyFrom(pLLVMIntrinsicCallOp);
      })

      .Case<cir::LLrintOp>([instID, pInst, pModuleID, &typeCache](cir::LLrintOp op) {
        protocir::CIRLLrintOp pLLrintOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_l_lrint_op()->CopyFrom(pLLrintOp);
      })

      .Case<cir::LLroundOp>([instID, pInst, pModuleID, &typeCache](cir::LLroundOp op) {
        protocir::CIRLLroundOp pLLroundOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_l_lround_op()->CopyFrom(pLLroundOp);
      })

      .Case<cir::LabelOp>([instID, pInst, pModuleID, &typeCache](cir::LabelOp op) {
        protocir::CIRLabelOp pLabelOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_label_op()->CopyFrom(pLabelOp);
      })

      .Case<cir::LoadOp>([instID, pInst, pModuleID, &typeCache](cir::LoadOp op) {
        protocir::CIRLoadOp pLoadOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_load_op()->CopyFrom(pLoadOp);
      })

      .Case<cir::Log10Op>([instID, pInst, pModuleID, &typeCache](cir::Log10Op op) {
        protocir::CIRLog10Op pLog10Op;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_log10_op()->CopyFrom(pLog10Op);
      })

      .Case<cir::Log2Op>([instID, pInst, pModuleID, &typeCache](cir::Log2Op op) {
        protocir::CIRLog2Op pLog2Op;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_log2_op()->CopyFrom(pLog2Op);
      })

      .Case<cir::LogOp>([instID, pInst, pModuleID, &typeCache](cir::LogOp op) {
        protocir::CIRLogOp pLogOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_log_op()->CopyFrom(pLogOp);
      })

      .Case<cir::LrintOp>([instID, pInst, pModuleID, &typeCache](cir::LrintOp op) {
        protocir::CIRLrintOp pLrintOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_lrint_op()->CopyFrom(pLrintOp);
      })

      .Case<cir::LroundOp>([instID, pInst, pModuleID, &typeCache](cir::LroundOp op) {
        protocir::CIRLroundOp pLroundOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_lround_op()->CopyFrom(pLroundOp);
      })

      .Case<cir::MemChrOp>([instID, pInst, pModuleID, &typeCache](cir::MemChrOp op) {
        protocir::CIRMemChrOp pMemChrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_chr_op()->CopyFrom(pMemChrOp);
      })

      .Case<cir::MemCpyInlineOp>([instID, pInst, pModuleID, &typeCache](cir::MemCpyInlineOp op) {
        protocir::CIRMemCpyInlineOp pMemCpyInlineOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_cpy_inline_op()->CopyFrom(pMemCpyInlineOp);
      })

      .Case<cir::MemCpyOp>([instID, pInst, pModuleID, &typeCache](cir::MemCpyOp op) {
        protocir::CIRMemCpyOp pMemCpyOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_cpy_op()->CopyFrom(pMemCpyOp);
      })

      .Case<cir::MemMoveOp>([instID, pInst, pModuleID, &typeCache](cir::MemMoveOp op) {
        protocir::CIRMemMoveOp pMemMoveOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_move_op()->CopyFrom(pMemMoveOp);
      })

      .Case<cir::MemSetInlineOp>([instID, pInst, pModuleID, &typeCache](cir::MemSetInlineOp op) {
        protocir::CIRMemSetInlineOp pMemSetInlineOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_set_inline_op()->CopyFrom(pMemSetInlineOp);
      })

      .Case<cir::MemSetOp>([instID, pInst, pModuleID, &typeCache](cir::MemSetOp op) {
        protocir::CIRMemSetOp pMemSetOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_mem_set_op()->CopyFrom(pMemSetOp);
      })

      .Case<cir::NearbyintOp>([instID, pInst, pModuleID, &typeCache](cir::NearbyintOp op) {
        protocir::CIRNearbyintOp pNearbyintOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_nearbyint_op()->CopyFrom(pNearbyintOp);
      })

      .Case<cir::ObjSizeOp>([instID, pInst, pModuleID, &typeCache](cir::ObjSizeOp op) {
        protocir::CIRObjSizeOp pObjSizeOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_obj_size_op()->CopyFrom(pObjSizeOp);
      })

      .Case<cir::PowOp>([instID, pInst, pModuleID, &typeCache](cir::PowOp op) {
        protocir::CIRPowOp pPowOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_pow_op()->CopyFrom(pPowOp);
      })

      .Case<cir::PrefetchOp>([instID, pInst, pModuleID, &typeCache](cir::PrefetchOp op) {
        protocir::CIRPrefetchOp pPrefetchOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_prefetch_op()->CopyFrom(pPrefetchOp);
      })

      .Case<cir::PtrDiffOp>([instID, pInst, pModuleID, &typeCache](cir::PtrDiffOp op) {
        protocir::CIRPtrDiffOp pPtrDiffOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_ptr_diff_op()->CopyFrom(pPtrDiffOp);
      })

      .Case<cir::PtrMaskOp>([instID, pInst, pModuleID, &typeCache](cir::PtrMaskOp op) {
        protocir::CIRPtrMaskOp pPtrMaskOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_ptr_mask_op()->CopyFrom(pPtrMaskOp);
      })

      .Case<cir::PtrStrideOp>([instID, pInst, pModuleID, &typeCache](cir::PtrStrideOp op) {
        protocir::CIRPtrStrideOp pPtrStrideOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_ptr_stride_op()->CopyFrom(pPtrStrideOp);
      })

      .Case<cir::ResumeOp>([instID, pInst, pModuleID, &typeCache](cir::ResumeOp op) {
        protocir::CIRResumeOp pResumeOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_resume_op()->CopyFrom(pResumeOp);
      })

      .Case<cir::ReturnAddrOp>([instID, pInst, pModuleID, &typeCache](cir::ReturnAddrOp op) {
        protocir::CIRReturnAddrOp pReturnAddrOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_return_addr_op()->CopyFrom(pReturnAddrOp);
      })

      .Case<cir::ReturnOp>([instID, pInst, pModuleID, &typeCache](cir::ReturnOp op) {
        protocir::CIRReturnOp pReturnOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_return_op()->CopyFrom(pReturnOp);
      })

      .Case<cir::RintOp>([instID, pInst, pModuleID, &typeCache](cir::RintOp op) {
        protocir::CIRRintOp pRintOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_rint_op()->CopyFrom(pRintOp);
      })

      .Case<cir::RotateOp>([instID, pInst, pModuleID, &typeCache](cir::RotateOp op) {
        protocir::CIRRotateOp pRotateOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_rotate_op()->CopyFrom(pRotateOp);
      })

      .Case<cir::RoundOp>([instID, pInst, pModuleID, &typeCache](cir::RoundOp op) {
        protocir::CIRRoundOp pRoundOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_round_op()->CopyFrom(pRoundOp);
      })

      .Case<cir::ScopeOp>([instID, pInst, pModuleID, &typeCache](cir::ScopeOp op) {
        protocir::CIRScopeOp pScopeOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_scope_op()->CopyFrom(pScopeOp);
      })

      .Case<cir::SelectOp>([instID, pInst, pModuleID, &typeCache](cir::SelectOp op) {
        protocir::CIRSelectOp pSelectOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_select_op()->CopyFrom(pSelectOp);
      })

      .Case<cir::SetBitfieldOp>([instID, pInst, pModuleID, &typeCache](cir::SetBitfieldOp op) {
        protocir::CIRSetBitfieldOp pSetBitfieldOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_set_bitfield_op()->CopyFrom(pSetBitfieldOp);
      })

      .Case<cir::ShiftOp>([instID, pInst, pModuleID, &typeCache](cir::ShiftOp op) {
        protocir::CIRShiftOp pShiftOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_shift_op()->CopyFrom(pShiftOp);
      })

      .Case<cir::SignBitOp>([instID, pInst, pModuleID, &typeCache](cir::SignBitOp op) {
        protocir::CIRSignBitOp pSignBitOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_sign_bit_op()->CopyFrom(pSignBitOp);
      })

      .Case<cir::SinOp>([instID, pInst, pModuleID, &typeCache](cir::SinOp op) {
        protocir::CIRSinOp pSinOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_sin_op()->CopyFrom(pSinOp);
      })

      .Case<cir::SqrtOp>([instID, pInst, pModuleID, &typeCache](cir::SqrtOp op) {
        protocir::CIRSqrtOp pSqrtOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_sqrt_op()->CopyFrom(pSqrtOp);
      })

      .Case<cir::StackRestoreOp>([instID, pInst, pModuleID, &typeCache](cir::StackRestoreOp op) {
        protocir::CIRStackRestoreOp pStackRestoreOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_stack_restore_op()->CopyFrom(pStackRestoreOp);
      })

      .Case<cir::StackSaveOp>([instID, pInst, pModuleID, &typeCache](cir::StackSaveOp op) {
        protocir::CIRStackSaveOp pStackSaveOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_stack_save_op()->CopyFrom(pStackSaveOp);
      })

      .Case<cir::StdFindOp>([instID, pInst, pModuleID, &typeCache](cir::StdFindOp op) {
        protocir::CIRStdFindOp pStdFindOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_std_find_op()->CopyFrom(pStdFindOp);
      })

      .Case<cir::StoreOp>([instID, pInst, pModuleID, &typeCache](cir::StoreOp op) {
        protocir::CIRStoreOp pStoreOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_store_op()->CopyFrom(pStoreOp);
      })

      .Case<cir::SwitchFlatOp>([instID, pInst, pModuleID, &typeCache](cir::SwitchFlatOp op) {
        protocir::CIRSwitchFlatOp pSwitchFlatOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_switch_flat_op()->CopyFrom(pSwitchFlatOp);
      })

      .Case<cir::SwitchOp>([instID, pInst, pModuleID, &typeCache](cir::SwitchOp op) {
        protocir::CIRSwitchOp pSwitchOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_switch_op()->CopyFrom(pSwitchOp);
      })

      .Case<cir::TernaryOp>([instID, pInst, pModuleID, &typeCache](cir::TernaryOp op) {
        protocir::CIRTernaryOp pTernaryOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_ternary_op()->CopyFrom(pTernaryOp);
      })

      .Case<cir::ThrowOp>([instID, pInst, pModuleID, &typeCache](cir::ThrowOp op) {
        protocir::CIRThrowOp pThrowOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_throw_op()->CopyFrom(pThrowOp);
      })

      .Case<cir::TrapOp>([instID, pInst, pModuleID, &typeCache](cir::TrapOp op) {
        protocir::CIRTrapOp pTrapOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_trap_op()->CopyFrom(pTrapOp);
      })

      .Case<cir::TruncOp>([instID, pInst, pModuleID, &typeCache](cir::TruncOp op) {
        protocir::CIRTruncOp pTruncOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_trunc_op()->CopyFrom(pTruncOp);
      })

      .Case<cir::TryCallOp>([instID, pInst, pModuleID, &typeCache](cir::TryCallOp op) {
        protocir::CIRTryCallOp pTryCallOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_try_call_op()->CopyFrom(pTryCallOp);
      })

      .Case<cir::TryOp>([instID, pInst, pModuleID, &typeCache](cir::TryOp op) {
        protocir::CIRTryOp pTryOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_try_op()->CopyFrom(pTryOp);
      })

      .Case<cir::UnaryOp>([instID, pInst, pModuleID, &typeCache](cir::UnaryOp op) {
        protocir::CIRUnaryOp pUnaryOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_unary_op()->CopyFrom(pUnaryOp);
      })

      .Case<cir::UnreachableOp>([instID, pInst, pModuleID, &typeCache](cir::UnreachableOp op) {
        protocir::CIRUnreachableOp pUnreachableOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_unreachable_op()->CopyFrom(pUnreachableOp);
      })

      .Case<cir::VAArgOp>([instID, pInst, pModuleID, &typeCache](cir::VAArgOp op) {
        protocir::CIRVAArgOp pVAArgOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_va_arg_op()->CopyFrom(pVAArgOp);
      })

      .Case<cir::VACopyOp>([instID, pInst, pModuleID, &typeCache](cir::VACopyOp op) {
        protocir::CIRVACopyOp pVACopyOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_va_copy_op()->CopyFrom(pVACopyOp);
      })

      .Case<cir::VAEndOp>([instID, pInst, pModuleID, &typeCache](cir::VAEndOp op) {
        protocir::CIRVAEndOp pVAEndOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_va_end_op()->CopyFrom(pVAEndOp);
      })

      .Case<cir::VAStartOp>([instID, pInst, pModuleID, &typeCache](cir::VAStartOp op) {
        protocir::CIRVAStartOp pVAStartOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_va_start_op()->CopyFrom(pVAStartOp);
      })

      .Case<cir::VTTAddrPointOp>([instID, pInst, pModuleID, &typeCache](cir::VTTAddrPointOp op) {
        protocir::CIRVTTAddrPointOp pVTTAddrPointOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vtt_addr_point_op()->CopyFrom(pVTTAddrPointOp);
      })

      .Case<cir::VTableAddrPointOp>([instID, pInst, pModuleID, &typeCache](cir::VTableAddrPointOp op) {
        protocir::CIRVTableAddrPointOp pVTableAddrPointOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_v_table_addr_point_op()->CopyFrom(pVTableAddrPointOp);
      })

      .Case<cir::VecCmpOp>([instID, pInst, pModuleID, &typeCache](cir::VecCmpOp op) {
        protocir::CIRVecCmpOp pVecCmpOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_cmp_op()->CopyFrom(pVecCmpOp);
      })

      .Case<cir::VecCreateOp>([instID, pInst, pModuleID, &typeCache](cir::VecCreateOp op) {
        protocir::CIRVecCreateOp pVecCreateOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_create_op()->CopyFrom(pVecCreateOp);
      })

      .Case<cir::VecExtractOp>([instID, pInst, pModuleID, &typeCache](cir::VecExtractOp op) {
        protocir::CIRVecExtractOp pVecExtractOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_extract_op()->CopyFrom(pVecExtractOp);
      })

      .Case<cir::VecInsertOp>([instID, pInst, pModuleID, &typeCache](cir::VecInsertOp op) {
        protocir::CIRVecInsertOp pVecInsertOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_insert_op()->CopyFrom(pVecInsertOp);
      })

      .Case<cir::VecShuffleDynamicOp>([instID, pInst, pModuleID, &typeCache](cir::VecShuffleDynamicOp op) {
        protocir::CIRVecShuffleDynamicOp pVecShuffleDynamicOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_shuffle_dynamic_op()->CopyFrom(pVecShuffleDynamicOp);
      })

      .Case<cir::VecShuffleOp>([instID, pInst, pModuleID, &typeCache](cir::VecShuffleOp op) {
        protocir::CIRVecShuffleOp pVecShuffleOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_shuffle_op()->CopyFrom(pVecShuffleOp);
      })

      .Case<cir::VecSplatOp>([instID, pInst, pModuleID, &typeCache](cir::VecSplatOp op) {
        protocir::CIRVecSplatOp pVecSplatOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_splat_op()->CopyFrom(pVecSplatOp);
      })

      .Case<cir::VecTernaryOp>([instID, pInst, pModuleID, &typeCache](cir::VecTernaryOp op) {
        protocir::CIRVecTernaryOp pVecTernaryOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_vec_ternary_op()->CopyFrom(pVecTernaryOp);
      })

      .Case<cir::WhileOp>([instID, pInst, pModuleID, &typeCache](cir::WhileOp op) {
        protocir::CIRWhileOp pWhileOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_while_op()->CopyFrom(pWhileOp);
      })

      .Case<cir::YieldOp>([instID, pInst, pModuleID, &typeCache](cir::YieldOp op) {
        protocir::CIRYieldOp pYieldOp;
        pInst->mutable_base()->set_id(instID);

        pInst->mutable_yield_op()->CopyFrom(pYieldOp);
      })

      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });

}
