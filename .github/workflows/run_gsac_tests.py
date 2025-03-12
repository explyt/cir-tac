#!/usr/bin/python3

import os
import sys
import filecmp

from utils import *


CIR_ORIGINAL = "test.s"
DESERIALIZED_FILE = "test.cir"


def run_and_check_test_result(test_path, cir_tac_path):
    # preparing files for the test run to have a clean state
    remove_if_exists(CIR_ORIGINAL)
    remove_if_exists(DESERIALIZED_FILE)

    test_res = run_test(cir_tac_path, test_path)

    if test_res != TestResult.Success:
        return False

    if not os.path.exists(CIR_ORIGINAL):
        print("Test succeeded but no original cir file found!")
        return False
    if not os.path.exists(DESERIALIZED_FILE):
        print("Test succeeded but no deserialized file found!")
        return False

    return filecmp.cmp(DESERIALIZED_FILE, CIR_ORIGINAL)


def main():
    if len(sys.argv) != 3:
        print("Expected paths to cir-tac and GSAC directories!")
        return -1

    cir_tac_path = sys.argv[1]
    gsac_path = sys.argv[2]
    test_files = get_test_paths(gsac_path)

    for test in test_files:
        print("Testing file [{0}]\n".format(test))
        if not run_and_check_test_result(test, cir_tac_path):
            print("Failure!")
            return 1
        print("Success!\n")

    return 0


if __name__ == "__main__":
    exit(main())
