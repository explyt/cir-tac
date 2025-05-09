// -*- C++ -*-
#pragma once

#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/TableGen/Class.h>

#include <iterator>
#include <optional>
#include <string>
#include <vector>

using mlir::raw_indented_ostream;
using namespace mlir::tblgen;
using namespace mlir;
using namespace llvm;

namespace ctgen {

struct PrivateField {
  std::string typ;
  std::string name;
  std::string init;
  bool isCtrParam;
};

struct HeaderInfo {
  std::string open;
  std::string close;

  HeaderInfo(std::string open, std::string close) : open(open), close(close) {}
};

struct CppTypeInfo {
  std::string factualType;
  std::string namedType;
};

struct SwitchCase {
  std::string cppType;
  std::string protoType;
  std::string caseValue;
  std::string caseBody;
  std::string translatorBody;
};

class CppSwitchSource {
protected:
  std::string funcName;
  std::string className;
  std::vector<SwitchCase> cases;
  std::vector<PrivateField> fields;

  CppTypeInfo resTy;

  std::string inputName;
  std::string serName;

  HeaderInfo declHeader;
  HeaderInfo defHeader;

  std::optional<std::string> preCaseBody;
  std::optional<std::string> postCaseBody;

  Class internalClass;

  virtual void dumpSwitchFunc(raw_indented_ostream &os) = 0;

  virtual Method *addMethod(std::string methodName, std::string returnType,
                            llvm::ArrayRef<MethodParameter> params) = 0;

  virtual Method *addTranslatorMethod(std::string protoType,
                                      std::string cppType,
                                      std::string methodName) = 0;

  void printCodeBlock(raw_indented_ostream &os, std::string code);

  void genCtr();

  void genClass();

  void addField(PrivateField f) { fields.push_back(f); }

  void addCase(SwitchCase c) { cases.push_back(c); }

  void addCase(std::string cpp, std::string proto, std::string val,
               std::string body, std::string translator) {
    addCase({cpp, proto, val, body, translator});
  }

  CppSwitchSource(std::string funcName, std::string className, CppTypeInfo ret,
                  std::string inputName, std::string declHedOpen,
                  std::string declHedClose, std::string defHedOpen,
                  std::string defHedClose)
      : funcName(funcName), className(className), resTy(ret),
        inputName(inputName), declHeader(declHedOpen, declHedClose),
        defHeader(defHedOpen, defHedClose), internalClass(className) {
    serName = convertToCamelFromSnakeCase(formatv("p_{0}", inputName).str());
  }

public:
  void addField(std::string typ, std::string name, std::string init) {
    fields.push_back({typ, name, init, /*isCtrParam=*/false});
  }

  void addField(std::string typ, std::string name) {
    fields.push_back({typ, name, name, /*isCtrParam=*/true});
  }

  void addPreCaseBody(std::string body) { preCaseBody = body; }

  void addPostCaseBody(std::string body) { postCaseBody = body; }

  void addHelperMethod(std::string methodName,
                       llvm::ArrayRef<MethodParameter> params,
                       std::string returnType, std::string methodBody) {
    auto &md = addMethod(methodName, returnType, params)->body();
    printCodeBlock(md.getStream(), methodBody);
  }

  inline void addHelperMethod(std::string methodName, MethodParameter param,
                              std::string returnType, std::string methodBody) {
    llvm::SmallVector<MethodParameter, 1> singleParam{std::move(param)};
    addHelperMethod(methodName, singleParam, returnType, methodBody);
  }

  void dumpDecl(llvm::raw_ostream &os) {
    os << declHeader.open << "\n";
    genClass();
    internalClass.finalize();
    internalClass.writeDeclTo(os);
    os << declHeader.close;
  }

  void dumpDef(llvm::raw_ostream &os) {
    os << defHeader.open << "\n";
    genClass();
    internalClass.finalize();
    internalClass.writeDefTo(os);
    os << defHeader.close;
  }

  virtual ~CppSwitchSource() = default;
};

class CppProtoSerializer : public CppSwitchSource {
private:
  void dumpSwitchFunc(raw_indented_ostream &os) override;
  Method *addMethod(std::string methodName, std::string returnType,
                    llvm::ArrayRef<MethodParameter> params) override;
  Method *addTranslatorMethod(std::string protoType, std::string cppType,
                              std::string methodName) override;

  const char *getStandardCaseBody() {
    return "auto serialized = serialize{0}({1});\n"
           "*{2}.mutable_{3}() = serialized;\n";
  }

public:
  CppProtoSerializer(std::string className, CppTypeInfo ret,
                     std::string inputName, std::string declHedOpen,
                     std::string declHedClose, std::string defHedOpen,
                     std::string defHedClose)
      : CppSwitchSource("serialize", className, ret, inputName, declHedOpen,
                        declHedClose, defHedOpen, defHedClose) {}

  void addStandardCase(std::string typ, std::string typName,
                       std::string translator) {
    auto snakeTypName = llvm::convertToSnakeFromCamelCase(typName);
    auto bodyFormat = getStandardCaseBody();
    auto caseBody =
        formatv(bodyFormat, typName, inputName, serName, snakeTypName);
    addCase(typ, typName, typName, caseBody, translator);
  }
};

class CppProtoDeserializer : public CppSwitchSource {
private:
  std::string deserName;
  std::string switchExpr;
  std::string switchParam;
  llvm::ArrayRef<MethodParameter> funcParams;

  std::string paramsCall;

  void dumpSwitchFunc(raw_indented_ostream &os) override;
  Method *addMethod(std::string methodName, std::string returnType,
                    llvm::ArrayRef<MethodParameter> params) override;
  Method *addTranslatorMethod(std::string protoType, std::string cppType,
                              std::string methodName) override;

  const char *standardCaseBody = "return deserialize{0}({3}{1}.{2}());\n";

  const char *getStandardCaseBody() { return standardCaseBody; }

public:
  void setStandardCaseBody(const char *newBody) { standardCaseBody = newBody; }

  CppProtoDeserializer(std::string className, CppTypeInfo ret,
                       std::string switchParam, std::string inputName,
                       std::string declHedOpen, std::string declHedClose,
                       std::string defHedOpen, std::string defHedClose,
                       llvm::ArrayRef<MethodParameter> inputParams)
      : CppSwitchSource("deserialize", className, ret, inputName, declHedOpen,
                        declHedClose, defHedOpen, defHedClose),
        switchParam(switchParam) {
    deserName = formatv("{0}Deser", inputName);
    auto snakeSwitchParam = llvm::convertToSnakeFromCamelCase(switchParam);
    switchExpr = formatv("{0}.{1}_case()", inputName, snakeSwitchParam);
    funcParams = inputParams;

    paramsCall = "";
    for (auto param : funcParams) {
      paramsCall += param.getName();
      paramsCall += ", ";
    }
  }

  void addStandardCase(std::string typ, std::string cppNamespace,
                       std::string typName, std::string translator) {
    auto snakeName = llvm::convertToSnakeFromCamelCase(typName);
    typName = formatv("{0}{1}", cppNamespace, typName);
    auto caseBody = formatv(getStandardCaseBody(), typName, inputName,
                            snakeName, paramsCall);
    // protobuf converts uppercase words to normal ones with first upper letter
    // converting first to snake and then to camel imitates this behaviour
    auto protoName = llvm::convertToCamelFromSnakeCase(snakeName, true);
    auto caseValue =
        formatv("{0}::{1}Case::k{2}", resTy.namedType, switchParam, protoName)
            .str();
    addCase(typ, typName, caseValue, caseBody, translator);
  }
};

} // namespace ctgen
