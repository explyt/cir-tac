#include "CXX/CIRType.h"

#include "CAPI/CIRType.h"
#include "CAPI/CIRTypeAPI.h"

const char *CIRTypeGetName(struct CIRTypeRef ref) {
  auto cirType = CIRType::fromRef(ref);
  return cirType.getName();
}

size_t CIRTypeGetSize(struct CIRTypeRef ref) {
  auto type = CIRType::fromRef(ref);
  return type.size();
}
