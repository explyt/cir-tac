if (NOT LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS Support)
endif()

set(LIBS
    MLIRTableGen
    LLVMTableGen
    MLIRSupport
    LLVMSupport
)

add_tablegen(cir-tblgen CIR
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    EXPORT LLVM
    EXPORT MLIR

    cir-tblgen.cpp
    PassCAPIGen.cpp
)

target_link_libraries(cir-tblgen PRIVATE ${LIBS})
