#!/usr/bin/python3

import os
import sys
import filecmp

from utils import *


CIR_ORIGINAL = "test.s"
DESERIALIZED_FILE = "test.cir"


def run_and_check_test_result(test_path, cir_tac_path, clang):
    # preparing files for the test run to have a clean state
    remove_if_exists(CIR_ORIGINAL)
    remove_if_exists(DESERIALIZED_FILE)
    remove_if_exists("test.out")

    test_res = run_test(cir_tac_path, test_path, enable_output=True, custom_clang=clang)

    if test_res != TestResult.Success:
        print("Failed to run test! Received result: {0}".format(test_res))
        return False

    if not os.path.exists(CIR_ORIGINAL):
        print("Test succeeded but no original cir file found!")
        return False
    if not os.path.exists(DESERIALIZED_FILE):
        print("Test succeeded but no deserialized file found!")
        return False

    return filecmp.cmp(DESERIALIZED_FILE, CIR_ORIGINAL)


def main():
    argc = len(sys.argv)
    if argc < 3 or argc > 4:
        print("Expected paths to cir-tac and GSAC directories, optionally to clang!")
        return -1

    cir_tac_path = sys.argv[1]
    gsac_path = sys.argv[2]

    clang = "clang" if argc == 3 else os.path.expanduser(sys.argv[3])

    test_files = get_test_paths(gsac_path)

    for test in test_files:
        print("Testing file [{0}]\n".format(test))
        if not run_and_check_test_result(test, cir_tac_path, clang):
            print("Failure!")
            return 1
        print("Success!\n")

    return 0


if __name__ == "__main__":
    exit(main())
