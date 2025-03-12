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


def run_test(cir_tac_path, test_path):
    cir_tac_path = os.path.expanduser(cir_tac_path)
    script_path = os.path.join(cir_tac_path, "run_ser_deser_cycle.py")
    test_cmd = "{2} \"{0}\" \"{1}\" > test.out".format(cir_tac_path, test_path, script_path)
    res = subprocess.run(test_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True).returncode
    return TestResult(res)


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
