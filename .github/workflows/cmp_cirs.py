#!/usr/bin/python3

import filecmp
import os
import sys

import utils


def main():
    if len(sys.argv) != 2:
        print("Expected path to directory containing original and serialized cirs!")
        return -1
    test_path = os.path.expanduser(sys.argv[1])
    test_files = utils.get_test_paths(test_path, "kt.cir")

    res = 0
    for test in test_files:
        print("Checking file [{0}]\n".format(test))
        utils.filter_ast_attrs(test)
        original = test[:-6] + "cir"
        if not os.path.exists(original):
            print("Could not find original .cir for [{0}]!".format(test))
            res = 1
            continue
        if not filecmp.cmp(test, original):
            print("Failure! Files are different\n\n")
            res = 1
            continue
        print("Success!\n\n")
    if res == 1:
        print("One or more tests failed!\n")
    return res


if __name__ == "__main__":
    exit(main())
