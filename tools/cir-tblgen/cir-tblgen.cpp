#include "mlir/TableGen/GenInfo.h"
#include "Tools/cir-tblgen/CirTblgenMain.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace cir;

// Generator that prints records.
mlir::GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

int main(int argc, char **argv) { return CirTblgenMain(argc, argv); }
