#include "cir-tac/Serializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;

void Serializer::serializeOperation(mlir::Operation &inst,
                                    protocir::CIROp *pInst,
                                    protocir::CIRModuleID pModuleID,
                                    TypeCache &typeCache,
                                    OperationCache &opCache,
                                    BlockCache &blockCache,
                                    FunctionCache &functionCache

) {
  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
      .Case<cir::AllocaOp>(
          [instID, pInst, pModuleID, &typeCache](cir::AllocaOp op) {
            protocir::CIRAllocaOp pAllocaOp;
            pAllocaOp.mutable_base()->set_id(instID);
            protocir::CIRTypeID pTypeID;
            pTypeID.mutable_module_id()->CopyFrom(pModuleID);
            pTypeID.set_id(internType(typeCache, op.getAllocaType()));
            pAllocaOp.mutable_alloca_type()->CopyFrom(pTypeID);
            if (op.getAlignment().has_value()) {
              pAllocaOp.set_alignment(op.getAlignment().value());
            }
            pInst->mutable_alloca()->CopyFrom(pAllocaOp);
          })
      .Case<cir::BinOp>([instID, pInst, pModuleID, &typeCache](cir::BinOp op) {
        protocir::CIRBinOp pBinOp;
        pBinOp.mutable_base()->set_id(instID);
        pInst->mutable_bin()->CopyFrom(pBinOp);
      })
      .Case<cir::LoadOp>([instID, pInst, pModuleID, &typeCache,
                          &opCache](cir::LoadOp op) {
        protocir::CIRLoadOp pLoadOp;
        pLoadOp.mutable_base()->set_id(instID);
        auto addressID = internOperation(opCache, op.getAddr().getDefiningOp());
        pLoadOp.mutable_address()->set_id(addressID);
        auto resultTypeID = internType(typeCache, op.getResult().getType());
        pLoadOp.mutable_result_type()->set_id(resultTypeID);
        pInst->mutable_load()->CopyFrom(pLoadOp);
      })
      .Case<cir::StoreOp>([instID, pInst, pModuleID, &typeCache,
                           &opCache](cir::StoreOp op) {
        protocir::CIRStoreOp pStoreOp;
        pStoreOp.mutable_base()->set_id(instID);
        auto addressID = internOperation(opCache, op.getAddr().getDefiningOp());
        pStoreOp.mutable_address()->set_id(addressID);
        auto valueID = internOperation(opCache, op.getAddr().getDefiningOp());
        pStoreOp.mutable_value()->set_id(valueID);
        pInst->mutable_store()->CopyFrom(pStoreOp);
      })
      .Case<cir::ConstantOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::ConstantOp op) {
            protocir::CIRConstantOp pConstantOp;
            pConstantOp.mutable_base()->set_id(instID);
            auto resultTypeID = internType(typeCache, op.getResult().getType());
            pConstantOp.mutable_result_type()->set_id(resultTypeID);
            pInst->mutable_constant()->CopyFrom(pConstantOp);
          })
      .Case<cir::CallOp>([instID, pInst, pModuleID, &typeCache, &opCache,
                          &functionCache](cir::CallOp op) {
        protocir::CIRCallOp pCallOp;
        pCallOp.mutable_base()->set_id(instID);
        if (op.getCallee().has_value()) {
          auto callee = op.getCallee().value();
          *pCallOp.mutable_callee() = callee.str();
        }
        for (auto arg : op.getArgOperands()) {
          auto pArg = pCallOp.add_arguments();
          auto argLine = internOperation(opCache, arg.getDefiningOp());
          pArg->set_id(argLine);
        }
        if (op.getNumResults() > 0) {
          auto resultTypeID = internType(typeCache, op.getResult().getType());
          pCallOp.mutable_result_type()->set_id(resultTypeID);
        }
        if (auto callable = op.resolveCallable()) {
          auto callableID =
              internFunction(functionCache, cast<cir::FuncOp>(callable));
        }
        pInst->mutable_call()->CopyFrom(pCallOp);
      })
      .Case<cir::ReturnOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::ReturnOp op) {
            protocir::CIRReturnOp pReturnOp;
            pReturnOp.mutable_base()->set_id(instID);
            for (auto input : op.getInput()) {
              auto inputID = internOperation(opCache, input.getDefiningOp());
              pReturnOp.add_input()->set_id(inputID);
            }
            pInst->mutable_return_()->CopyFrom(pReturnOp);
          })
      .Case<cir::GetGlobalOp>([instID, pInst, pModuleID, &typeCache,
                               &opCache](cir::GetGlobalOp op) {
        protocir::CIRGetGlobalOp pGetGlobalOp;
        pGetGlobalOp.mutable_base()->set_id(instID);
        *pGetGlobalOp.mutable_name() = op.getName().str();
        pInst->mutable_get_global()->CopyFrom(pGetGlobalOp);
      })
      .Case<cir::CastOp>(
          [instID, pInst, pModuleID, &typeCache, &opCache](cir::CastOp op) {
            protocir::CIRCastOp pCastOp;
            pCastOp.mutable_base()->set_id(instID);
            auto srcID = internOperation(opCache, op.getSrc().getDefiningOp());
            pCastOp.mutable_src()->set_id(srcID);
            auto resultTypeID = internType(typeCache, op.getResult().getType());
            pCastOp.mutable_result_type()->set_id(resultTypeID);
            pInst->mutable_cast()->CopyFrom(pCastOp);
          })
      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });
}
