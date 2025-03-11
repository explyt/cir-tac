#!/usr/bin/python3

import subprocess
import os
import sys
import filecmp

from pathlib import Path


CIR_ORIGINAL = "test.s"
DESERIALIZED_FILE = "test.cir"


def get_test_cmd_line(cir_tac_path, test_path):
    script_path = os.path.join(cir_tac_path, "run_ser_deser_cycle.py")
    return "{2} \"{0}\" \"{1}\" > test.out".format(cir_tac_path, test_path, script_path)


def filter_ast_attrs(file_path):
    with open(file_path, "r") as file:
        code = file.read().replace(" #cir.record.decl.ast", "")
    with open(file_path, "w") as file:
        file.write(code)


def get_test_paths(root):
    return [x.absolute().as_posix() for x in Path(root).rglob("*.c")]


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def run_test(test_path, cir_tac_path):
    # preparing files for the test run to have a clean state
    remove_if_exists(CIR_ORIGINAL)
    remove_if_exists(DESERIALIZED_FILE)

    test_cmd = get_test_cmd_line(cir_tac_path, test_path)
    subprocess.run(test_cmd, shell=True)

    # removing sometimes appearing empty ast attributes
    filter_ast_attrs(CIR_ORIGINAL)

    return filecmp.cmp(DESERIALIZED_FILE, CIR_ORIGINAL)


def main():
    if len(sys.argv) != 3:
        print("Expected paths to cir-tac and GSAC directories!")
        return -1

    cir_tac_path = os.path.expanduser(sys.argv[1])
    gsac_path = os.path.expanduser(sys.argv[2])
    test_files = get_test_paths(gsac_path)

    for test in test_files:
        print("Testing file [{0}]\n".format(test))
        if not run_test(test, cir_tac_path):
            print("Failure!")
            return 1
        print("Success!\n")

    return 0


if __name__ == "__main__":
    exit(main())
