#include "proto/model.pb.h"
#include "proto/type.pb.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Types.h>

using TypeCache = llvm::DenseMap<mlir::Type, std::string>;

using OperationCache = llvm::DenseMap<mlir::Operation *, uint64_t>;

using BlockCache = llvm::DenseMap<mlir::Block *, uint64_t>;

namespace protocir {
class Serializer {
public:
  static std::string internType(TypeCache &cache, mlir::Type type) {
    if (!cache.contains(type)) {
      std::string idStr;
      llvm::raw_string_ostream nameStream(idStr);
      type.print(nameStream);
      cache[type] = idStr;
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

  static protocir::CIROp serializeOperation(mlir::Operation &inst,
                                            protocir::CIRModuleID pModuleID,
                                            TypeCache &typeCache,
                                            OperationCache &opCache,
                                            BlockCache &blockCache);

  static protocir::CIRValue serializeValue(mlir::Value &value,
                                           protocir::CIRModuleID pModuleID,
                                           TypeCache &typeCache,
                                           OperationCache &opCache,
                                           BlockCache &blockCache);

  static protocir::CIRType serializeType(::mlir::Type &cirKind,
                                         protocir::CIRModuleID pModuleID,
                                         TypeCache &typeCache);

  static protocir::CIRRecordKind
  serializeRecordKind(::cir::StructType::RecordKind cirKind);
};

} // namespace protocir
