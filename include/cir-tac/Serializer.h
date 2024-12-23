#include "proto/model.pb.h"
#include "proto/type.pb.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Types.h>

using TypeCache = llvm::DenseMap<mlir::Type, uint64_t>;

using OperationCache = llvm::DenseMap<mlir::Operation *, uint64_t>;

using BlockCache = llvm::DenseMap<mlir::Block *, uint64_t>;

using FunctionCache = llvm::DenseMap<cir::FuncOp, uint64_t>;

namespace protocir {
class Serializer {
public:
  static uint64_t internType(TypeCache &cache, mlir::Type type) {
    if (!cache.contains(type)) {
      cache[type] = cache.size();
    }
    return cache.at(type);
  }

  static uint64_t internOperation(OperationCache &cache,
                                  mlir::Operation *operation) {
    if (!cache.contains(operation)) {
      cache[operation] = cache.size();
    }
    return cache.at(operation);
  }

  static uint64_t internBlock(BlockCache &cache, mlir::Block *block) {
    if (!cache.contains(block)) {
      cache[block] = cache.size();
    }
    return cache.at(block);
  }

  static uint64_t internFunction(FunctionCache &cache, cir::FuncOp operation) {
    if (!cache.contains(operation)) {
      cache[operation] = cache.size();
    }
    return cache.at(operation);
  }

  static void serializeOperation(mlir::Operation &inst, protocir::CIROp *pInst,
                                 protocir::CIRModuleID pModuleID,
                                 TypeCache &typeCache, OperationCache &opCache,
                                 BlockCache &blockCache,
                                 FunctionCache &functionCache);

  static protocir::CIRType serializeType(::mlir::Type &cirKind,
                                         TypeCache &typeCache);
};

} // namespace protocir
