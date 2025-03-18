#!/usr/bin/python3

import os
import subprocess

from pathlib import Path
from enum import Enum


class TestResult(Enum):
    Success = 0
    BuildError = 1
    SerializationError = 2
    DeserializationError = 3


def get_test_paths(root):
    root = os.path.expanduser(root)
    return [x.absolute().as_posix() for x in Path(root).rglob("*.c")]


def run_test(cir_tac_path, test_path, enable_output=False, custom_clang=None):
    cir_tac_path = os.path.expanduser(cir_tac_path)
    script_path = os.path.join(cir_tac_path, "scripts", "run_ser_deser_cycle.py")

    clang = "" if custom_clang is None else "\"{0}\" ".format(custom_clang)

    test_cmd = "{2} \"{0}\" \"{1}\" {3}> test.out".format(cir_tac_path, test_path, script_path, clang)
    kwargs = {}
    kwargs['shell'] = True
    if not enable_output:
        kwargs['stdin'] = subprocess.DEVNULL
        kwargs['stdout'] = subprocess.DEVNULL
    res = subprocess.run(test_cmd, **kwargs).returncode
    return TestResult(res)


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
