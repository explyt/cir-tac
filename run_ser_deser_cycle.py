#!/usr/bin/python3

import subprocess
import sys
from pathlib import Path


def create_file_name(name, ext):
    return "{0}.{1}".format(name, ext)


def run_translation_cmd(cmd, fr, to):
    subprocess.run("{0} {1} > {2}".format(cmd, fr, to), shell=True)


def main():
    test_name = "test" if len(sys.argv) != 2 else Path(sys.argv[1]).stem
    test_src = create_file_name(test_name, "cpp") if len(sys.argv) != 2 else sys.argv[1]
    test_cir = create_file_name(test_name, "s")
    test_ser = create_file_name(test_name, "proto")
    test_deser = create_file_name(test_name, "cir")
    run_translation_cmd("clang -S -Xclang -emit-cir-flat", test_src, test_cir)
    run_translation_cmd("tools/cir-ser-proto/cir-ser-proto", test_cir, test_ser)
    run_translation_cmd("tools/cir-deser-proto/cir-deser-proto", test_ser, test_deser)
    print("\n\nDIFF OUTPUT:\n\n")
    subprocess.run("diff {0} {1}".format(test_cir, test_deser), shell=True)

if __name__ == "__main__":
    main()
