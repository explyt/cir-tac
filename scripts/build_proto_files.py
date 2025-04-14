#!/usr/bin/python3

import os
import subprocess
import sys

import cir_tblgen_util


def get_include_args(clangir_path):
    clang_path = os.path.join(clangir_path, "clang", "include")
    mlir_path = os.path.join(clangir_path, "mlir", "include")
    return "-I {0} -I {1}".format(clang_path, mlir_path)


def get_td_path(clangir_path, td_file):
    td_path = os.path.join(
        clangir_path, "clang", "include", "clang", "CIR", "Dialect", "IR"
    )
    return os.path.join(td_path, td_file)


def run_tblgen_command(clangir_path, cir_tac, subcmd, td_file, result_path):
    tblgen_path = os.path.join(
        cir_tac, "build", "tools", "cir-tac-tblgen", "cir-tac-tblgen"
    )
    cmd = "{0} {1} --{2} {3} > {4}".format(
        tblgen_path,
        get_include_args(clangir_path),
        subcmd,
        get_td_path(clangir_path, td_file),
        result_path,
    )

    subprocess.run(cmd, check=True, shell=True)


def gen_file(clangir_path, cir_tac, result_dir, file_info: cir_tblgen_util.TblgenFileInfo):
    result_path = os.path.join(result_dir, file_info.path)
    subprocess.run("rm -f {0}".format(result_path), shell=True)
    run_tblgen_command(clangir_path, cir_tac, file_info.subcmd, file_info.td, result_path)


def main():
    argc = len(sys.argv)
    if not (5 == argc):
        print("Expected paths to clangir, cir-tac, repo directories"
              "and files type to generate!")
        return -1
    clangir = os.path.expanduser(sys.argv[1])
    cir_tac = os.path.expanduser(sys.argv[2])
    repo_root = os.path.expanduser(sys.argv[3])
    gen_type = sys.argv[4]
    if gen_type != "cpp" and gen_type != "kotlin":
        print("Gen type can be either cpp or kotlin!")
        return -5

    os.chdir(repo_root)

    if gen_type == "cpp":
        file_infos = cir_tblgen_util.get_tblgen_file_infos_cpp()
    if gen_type == "kotlin":
        file_infos = cir_tblgen_util.get_tblgen_file_infos_kotlin()
    for file_info in file_infos:
        gen_file(clangir, cir_tac, repo_root, file_info)


if __name__ == "__main__":
    exit(main())
