#!/usr/bin/python3

import filecmp
import os
import sys
import shutil

import utils

CIR_ORIGINAL = "test.s"
DESERIALIZED_FILE = "test.cir"
TEST_OUTPUT = "test.out"


def run_and_check_test_result(test_path, cir_tac_path, clang):
    # preparing files for the test run to have a clean state
    utils.remove_if_exists(CIR_ORIGINAL)
    utils.remove_if_exists(DESERIALIZED_FILE)
    utils.remove_if_exists(TEST_OUTPUT)

    test_res = utils.run_test(
        cir_tac_path, test_path, enable_output=True, custom_clang=clang
    )

    if test_res != utils.TestResult.Success:
        print("Failed to run test! Received result: {0}".format(test_res))
        return False

    if not os.path.exists(CIR_ORIGINAL):
        print("Test succeeded but no original cir file found!")
        return False
    if not os.path.exists(DESERIALIZED_FILE):
        print("Test succeeded but no deserialized file found!")
        return False

    return filecmp.cmp(DESERIALIZED_FILE, CIR_ORIGINAL)


def save_failed_test(test_path, gsac_path):
    test_relpath = os.path.relpath(test_path, gsac_path)
    test_name = test_relpath.replace(os.path.sep, "_")
    for test_outs in [CIR_ORIGINAL, DESERIALIZED_FILE, TEST_OUTPUT]:
        save_copy_name = "{0}_{1}".format(test_name, test_outs)
        save_copy_path = os.path.join("failures", save_copy_name)
        shutil.copyfile(test_outs, save_copy_path)


def main():
    argc = len(sys.argv)
    if argc < 3 or argc > 4:
        print("Expected paths to cir-tac and GSAC directories, optionally to clang!")
        return -1

    cir_tac_path = os.path.expanduser(sys.argv[1])
    gsac_path = os.path.expanduser(sys.argv[2])

    clang = "clang" if argc == 3 else os.path.expanduser(sys.argv[3])

    test_files = utils.get_test_paths(gsac_path)

    res = 0
    for test in test_files:
        print("Testing file [{0}]\n".format(test))
        if not run_and_check_test_result(test, cir_tac_path, clang):
            print("Failure! Saving test outputs...\n")
            save_failed_test(test, gsac_path)
            res = 1
        print("Success!\n")
    return res


if __name__ == "__main__":
    exit(main())
