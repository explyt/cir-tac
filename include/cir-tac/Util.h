#pragma once

#include "proto/setup.pb.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Types.h>

#include <string>

using namespace protocir;

class TypeCache {
public:
  TypeCache(MLIRModuleID moduleID) : moduleID(moduleID) {}

  MLIRTypeID getMLIRTypeID(mlir::Type type) {
    if (!cache_.contains(type)) {
      std::string name;
      llvm::raw_string_ostream nameStream(name);
      type.print(nameStream);
      MLIRTypeID typeID;
      *typeID.mutable_id() = name;
      *typeID.mutable_module_id() = moduleID;
      cache_[type] = typeID;
    }
    return cache_.at(type);
  }

  llvm::DenseMap<mlir::Type, MLIRTypeID> &map() { return cache_; }

private:
  MLIRModuleID moduleID;
  llvm::DenseMap<mlir::Type, MLIRTypeID> cache_;
};

class OpCache {
public:
  MLIROpID getMLIROpID(mlir::Operation *operation) {
    if (!cache_.contains(operation)) {
      MLIROpID opID;
      opID.set_id(cache_.size());
      cache_[operation] = opID;
    }
    return cache_.at(operation);
  }

private:
  llvm::DenseMap<mlir::Operation *, MLIROpID> cache_;
};

class BlockCache {
public:
  MLIRBlockID getMLIRBlockID(mlir::Block *block) {
    if (!cache_.contains(block)) {
      MLIRBlockID blockID;
      blockID.set_id(cache_.size());
      cache_[block] = blockID;
    }
    return cache_.at(block);
  }

private:
  llvm::DenseMap<mlir::Block *, MLIRBlockID> cache_;
};

inline std::string serializeAPInt(llvm::APInt i) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  i.print(stream, true);
  return result;
}

inline std::string serializeAPFloat(llvm::APFloat f) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  f.print(stream);
  return result;
}

inline std::string serializeStringRef(llvm::StringRef r) { return r.str(); }
