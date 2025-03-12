#!/usr/bin/python3

import subprocess
import sys
import os
from pathlib import Path


def create_file_name(name, ext):
    return "{0}.{1}".format(name, ext)


def run_translation_cmd(cmd, fr, to):
    return subprocess.run("{0} {1} > {2}".format(cmd, fr, to), shell=True).returncode == 0


def filter_ast_attrs(file_path):
    with open(file_path, "r") as file:
        code = file.read().replace(" #cir.record.decl.ast", "")
    with open(file_path, "w") as file:
        file.write(code)


def main():
    if len(sys.argv) != 3:
        print("Expected paths to cir-tac directory and ClangIR file!")
        return -1

    ser_tool_path = os.path.join(os.path.expanduser(sys.argv[1]),
                                 "build", "tools", "cir-ser-proto", "cir-ser-proto")
    des_tool_path = os.path.join(os.path.expanduser(sys.argv[1]),
                                 "build", "tools", "cir-deser-proto", "cir-deser-proto")

    test_src = sys.argv[2]
    test_name = "test"
    test_cir = create_file_name(test_name, "s")
    test_ser = create_file_name(test_name, "proto")
    test_deser = create_file_name(test_name, "cir")

    compile_cmd = "clang -S -Xclang -emit-cir-flat -o {1} {0}".format(test_src, test_cir)
    if subprocess.run(compile_cmd, shell=True).returncode != 0:
        print("Compile error!")
        return 1
    if not run_translation_cmd(ser_tool_path, test_cir, test_ser):
        print("Serialization error!")
        return 2
    if not run_translation_cmd(des_tool_path, test_ser, test_deser):
        print("Deserialization error!")
        return 3

    # removing sometimes appearing empty ast attributes
    filter_ast_attrs(test_cir)

    print("\n\nDIFF OUTPUT:\n\n")
    subprocess.run("diff {0} {1}".format(test_cir, test_deser), shell=True)


if __name__ == "__main__":
    exit(main())
