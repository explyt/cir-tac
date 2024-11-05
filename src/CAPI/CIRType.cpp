#include "CAPI/CIRType.h"
#include "CXX/CIRType.h"

const char *CIRTypeGetName(struct CIRTypeRef ref) {
  auto cirType = CIRType::fromRef(ref);
  return cirType.getName();
}

size_t CIRGetTypeSize(struct CIRTypeRef ref) {
  auto type = CIRType::fromRef(ref);
  return type.size();
}
