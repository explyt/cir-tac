#!/usr/bin/python3

import os

from enum import Enum
from dataclasses import dataclass


class CodeType(Enum):
    RawFile = 0
    CppProto = 1
    CppDecl = 2
    CppSrc = 3
    KotlinSrc = 4
    KotlinProto = 5
    KotlinBuilder = 6
    KotlinSerializer = 7


def id(file_name):
    return file_name


CPP_PROTO_PREFIX = ["proto"]
CPP_DECL_PREFIX = ["include", "cir-tac"]
CPP_SRC_PREFIX = ["src"]
KOTLIN_SRC_PREFIX = [
    "jacodb-api-cir", "src", "main", "kotlin", "org", "jacodb", "api", "cir", "cfg"
]
KOTLIN_PROTO_PREFIX = ["jacodb-core-cir", "src", "main", "proto"]
KOTLIN_BUILDER_PREFIX = [
    "jacodb-core-cir", "src", "main", "kotlin", "org", "jacodb", "impl", "cfg",
    "builder"
]
KOTLIN_SERIALIZER_PREFIX = [
    "jacodb-core-cir", "src", "main", "kotlin", "org", "jacodb", "impl", "cfg",
    "serializer", "tblgenerated"
]


MAP_TYPE_TO_PATH_PREFIX = {
    CodeType.RawFile: id,
    CodeType.CppProto: CPP_PROTO_PREFIX,
    CodeType.CppDecl: CPP_DECL_PREFIX,
    CodeType.CppSrc: CPP_SRC_PREFIX,
    CodeType.KotlinSrc: KOTLIN_SRC_PREFIX,
    CodeType.KotlinProto: KOTLIN_PROTO_PREFIX,
    CodeType.KotlinBuilder: KOTLIN_BUILDER_PREFIX,
    CodeType.KotlinSerializer: KOTLIN_SERIALIZER_PREFIX,
}


@dataclass
class TblgenFileInfo:
    subcmd: str
    td: str
    name: str
    typ: CodeType

    def __post_init__(self):
        self.path: str = os.path.join(*MAP_TYPE_TO_PATH_PREFIX[self.typ], self.name)


def get_subcmd_cpp_file_infos(subcmd, td_file, name, no_deser=False) -> list[TblgenFileInfo]:
    cpp_name = str.capitalize(name)

    files = []
    files.append(
        TblgenFileInfo(
            subcmd,
            td_file,
            "{0}.proto".format(name),
            CodeType.CppProto,
        )
    )
    files.append(
        TblgenFileInfo(
            "{0}-serializer-header".format(subcmd),
            td_file,
            "{0}Serializer.h".format(cpp_name),
            CodeType.CppDecl,
        )
    )
    files.append(
        TblgenFileInfo(
            "{0}-serializer-source".format(subcmd),
            td_file,
            "{0}Serializer.cpp".format(cpp_name),
            CodeType.CppSrc,
        )
    )

    if no_deser:
        return files

    files.append(
        TblgenFileInfo(
            "{0}-deserializer-header".format(subcmd),
            td_file,
            "{0}Deserializer.h".format(cpp_name),
            CodeType.CppDecl
        )
    )
    files.append(
        TblgenFileInfo(
            "{0}-deserializer-source".format(subcmd),
            td_file,
            "{0}Deserializer.cpp".format(cpp_name),
            CodeType.CppSrc
        )
    )
    return files


def get_subcmd_kotlin_file_infos(td_file, name, inc_proto=True, inc_builder=True) -> list[TblgenFileInfo]:
    files = []
    kotlin_name = str.capitalize(name)

    files.append(
        TblgenFileInfo(
            "gen-{0}-kotlin".format(name),
            td_file,
            "CIR{0}.kt".format(kotlin_name),
            CodeType.KotlinSrc
        )
    )

    if inc_builder:
        files.append(
            TblgenFileInfo(
                "gen-{0}-kotlin-builder".format(name),
                td_file,
                "{0}.kt".format(kotlin_name),
                CodeType.KotlinBuilder
            )
        )

    files.append(
        TblgenFileInfo(
            "gen-{0}-proto-serializer-kotlin".format(name),
            td_file,
            "{0}Serializer.kt".format(kotlin_name),
            CodeType.KotlinSerializer
        )
    )

    if inc_proto:
        files.append(
            TblgenFileInfo(
                "gen-{0}-proto".format(name),
                td_file,
                "{0}.proto".format(name),
                CodeType.KotlinProto
            )
        )
    return files


def get_tblgen_file_infos_cpp() -> list[TblgenFileInfo]:
    file_infos = []
    file_infos += get_subcmd_cpp_file_infos("gen-op-proto", "CIROps.td", "op")
    file_infos += get_subcmd_cpp_file_infos("gen-enum-proto", "CIROps.td", "enum")
    file_infos += get_subcmd_cpp_file_infos(
        "gen-type-proto", "MLIRCIRTypes.td", "type", no_deser=True
    )
    file_infos += get_subcmd_cpp_file_infos(
        "gen-attr-proto", "MLIRCIRAttrs.td", "attr"
    )
    return file_infos


def get_tblgen_file_infos_kotlin() -> list[TblgenFileInfo]:
    file_infos = []
    file_infos += get_subcmd_kotlin_file_infos("CIROps.td", "inst", inc_proto=False)
    file_infos += get_subcmd_kotlin_file_infos("CIROps.td", "expr", inc_proto=False)
    file_infos += get_subcmd_kotlin_file_infos("CIROps.td", "enum")
    file_infos += get_subcmd_kotlin_file_infos("MLIRCIRTypes.td", "type")
    file_infos += get_subcmd_kotlin_file_infos("MLIRCIRAttrs.td", "attr")
    file_infos.append(
        TblgenFileInfo(
            "gen-op-proto",
            "CIROps.td",
            "op.proto",
            CodeType.KotlinProto
        )
    )
    file_infos.append(
        TblgenFileInfo(
            "gen-op-proto-serializer-kotlin",
            "CIROps.td",
            "OpSerializer.kt",
            CodeType.KotlinSerializer
        )
    )
    return file_infos
