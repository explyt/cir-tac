/* Autogenerated by mlir-tblgen; don't manually edit. */
// clang-format off

#pragma once

#include "proto/enum.pb.h"

#include <clang/CIR/Dialect/IR/CIROpsEnums.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace protocir;

CIRAsmFlavor serializeCIRAsmFlavor(cir::AsmFlavor);
CIRAtomicFetchKind serializeCIRAtomicFetchKind(cir::AtomicFetchKind);
CIRAwaitKind serializeCIRAwaitKind(cir::AwaitKind);
CIRBinOpKind serializeCIRBinOpKind(cir::BinOpKind);
CIRBinOpOverflowKind serializeCIRBinOpOverflowKind(cir::BinOpOverflowKind);
CIRCallingConv serializeCIRCallingConv(cir::CallingConv);
CIRCaseOpKind serializeCIRCaseOpKind(cir::CaseOpKind);
CIRCastKind serializeCIRCastKind(cir::CastKind);
CIRCatchParamKind serializeCIRCatchParamKind(cir::CatchParamKind);
CIRCmpOpKind serializeCIRCmpOpKind(cir::CmpOpKind);
CIRCmpOrdering serializeCIRCmpOrdering(cir::CmpOrdering);
CIRComplexBinOpKind serializeCIRComplexBinOpKind(cir::ComplexBinOpKind);
CIRComplexRangeKind serializeCIRComplexRangeKind(cir::ComplexRangeKind);
CIRDynamicCastKind serializeCIRDynamicCastKind(cir::DynamicCastKind);
CIRGlobalLinkageKind serializeCIRGlobalLinkageKind(cir::GlobalLinkageKind);
CIRInlineKind serializeCIRInlineKind(cir::InlineKind);
CIRMemOrder serializeCIRMemOrder(cir::MemOrder);
CIRSignedOverflowBehavior serializeCIRSignedOverflowBehavior(cir::sob::SignedOverflowBehavior);
CIRSizeInfoType serializeCIRSizeInfoType(cir::SizeInfoType);
CIRSourceLanguage serializeCIRSourceLanguage(cir::SourceLanguage);
CIRTLSModel serializeCIRTLSModel(cir::TLS_Model);
CIRUnaryOpKind serializeCIRUnaryOpKind(cir::UnaryOpKind);
CIRVisibilityKind serializeCIRVisibilityKind(cir::VisibilityKind);
CIRRecordKind serializeCIRRecordKind(cir::StructType::RecordKind);
MLIRSignednessSemantics serializeMLIRSignednessSemantics(mlir::IntegerType::SignednessSemantics);

// clang-format on
