# CAPI #

This directory contains C API (CAPI) header files designed to interface with the Clang-IR infrastructure. These headers provide essential function definitions, data structures, and macros to enable smooth interaction between C-based applications and CIR components.

In general, API follows such convention:

* `CIR{Struct}.h` declares references to the `Struct` in clangir infrastructure
* `CIR{Struct}API.h` declares operations on the `CIR{Struct}.h`'s references
