#!/usr/bin/python3

import subprocess


def create_file_name(name, ext):
    return "{0}.{1}".format(name, ext)


def main():
    test_name = "test"
    test_src = create_file_name(test_name, "cpp")
    test_cir = create_file_name(test_name, "s")
    test_ser = create_file_name(test_name, "proto")
    test_deser = create_file_name(test_name, "cir")
    subprocess.run("clang -S -Xclang -emit-cir {0}".format(test_src), shell=True)
    subprocess.run("tools/cir-ser-proto/cir-ser-proto {0} > {1}".format(test_cir, test_ser), shell=True)
    subprocess.run("tools/cir-deser-proto/cir-deser-proto {0} > {1}".format(test_ser, test_deser), shell=True)

if __name__ == "__main__":
    main()
