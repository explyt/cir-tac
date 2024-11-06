# cir-tac #

## Description ##
CAPI library implementation for the CIR dialect of MLIR.

## Build ##

To build, firstly, you will need to build the [clangir](https://github.com/llvm/clangir)  (if you have not done so already). To do this, run the following commands (**not necessary in this directory**):

```bash
$ git clone git@github.com:llvm/clangir.git
$ cd clangir/llvm
$ mkdir build
$ cd build
$ cmake -GNinja \
$ -DLLVM_ENABLE_PROJECTS="clang;mlir" \ 
$ -DCLANG_ENABLE_CIR=ON ..
$ ninja
```

**Note**: On 06.11.2024 building with `ninja` is required to build `clangir`.

After that you can tun the following sequence of commands in this repository (here `PATH_TO_CLANGIR_BUILD` refers to build directory of clangir, for instance, above it will be `.../clangir/llvm/build`):

```bash
$ mkdir build
$ cd build
$ cmake -DCLANGIR_BUILD_DIR=${PATH_TO_CLANGIR_BUILD} .. && make
```

This will produce a static library called `CIRCAPI`, which contains the CAPI implementation.
