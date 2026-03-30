# injec_dh_tools.py
# -*- coding: utf-8 -*-
"""
从 InjecAgent 的 direct-harm 相关文件中抽取“工具目录”信息：
- user_cases.jsonl      → 用户工具（User Tools）定义
- attacker_cases_dh.jsonl → 直接危害攻击工具（Attacker Tools, direct harm）

后续会用这些 spec 去构建统一的 17 个用户工具 + direct-harm 攻击工具。
"""

import json
import ast
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# ===================== 1. 数据加载函数 =====================


def load_user_cases(path: str = "user_cases.jsonl") -> List[Dict[str, Any]]:
    """加载所有 user_cases（每一行一个 JSON）。"""
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(obj)
    return cases


def load_attacker_cases_dh(
    path: str = "attacker_cases_dh.jsonl",
) -> List[Dict[str, Any]]:
    """加载所有 direct-harm attacker cases（每一行一个 JSON）。"""
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(obj)
    return cases


# ===================== 2. 用户工具规格 =====================


@dataclass
class UserToolSpec:
    """
    对一个“用户工具”的抽象：

    - name: 工具名（如 "AmazonGetProductDetails"）
    - param_keys: 参数名列表（如 ["product_id"]）
    - example_params: 从 user_cases 里拿到的一份示例参数
    - response_template: 一条 Tool Response Template（里有 <Attacker Instruction> 占位符）
    """

    name: str
    param_keys: List[str]
    example_params: Dict[str, Any]
    response_template: str


def extract_user_tool_specs(
    user_cases: List[Dict[str, Any]],
) -> Dict[str, UserToolSpec]:
    """
    从 user_cases.jsonl 中抽取所有 User Tool 的参数信息和模板。
    """
    tool_params_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tool_templates: Dict[str, str] = {}

    for c in user_cases:
        tool_name = c["User Tool"]
        param_str = c.get("Tool Parameters", "")
        try:
            params = ast.literal_eval(param_str)
            if isinstance(params, dict):
                tool_params_raw[tool_name].append(params)
        except Exception:
            pass

        if tool_name not in tool_templates:
            tool_templates[tool_name] = c.get("Tool Response Template", "")

    specs: Dict[str, UserToolSpec] = {}

    for tool_name, plist in tool_params_raw.items():
        all_keys: Set[str] = set()
        for p in plist:
            all_keys.update(p.keys())
        example = plist[0] if plist else {}
        response_template = tool_templates.get(tool_name, "")

        specs[tool_name] = UserToolSpec(
            name=tool_name,
            param_keys=sorted(all_keys),
            example_params=example,
            response_template=response_template,
        )

    return specs


# ===================== 3. 攻击工具名字（direct harm） =====================


def extract_attacker_tool_names_dh(attacker_cases: List[Dict[str, Any]]) -> Set[str]:
    """
    从 attacker_cases_dh.jsonl 中提取所有攻击工具的名字。
    """
    names: Set[str] = set()
    for c in attacker_cases:
        for t in c.get("Attacker Tools", []):
            if t:
                names.add(t)
    return names


# ===================== 4. 简单 Demo：打印看看效果 =====================

if __name__ == "__main__":
    user_cases = load_user_cases("/home/Inject_defense/user_cases.jsonl")
    attacker_cases = load_attacker_cases_dh(
        "/home/Inject_defense/attacker_cases_dh.jsonl"
    )

    print(f"[INFO] user_cases 条数: {len(user_cases)}")
    print(f"[INFO] attacker_cases_dh 条数: {len(attacker_cases)}\n")

    user_tool_specs = extract_user_tool_specs(user_cases)
    print("=== Direct-Harm 子集中的 User Tools（工具名 + 参数） ===")
    for name, spec in sorted(user_tool_specs.items(), key=lambda x: x[0]):
        sig = ", ".join(spec.param_keys)
        print(f"- {name}({sig})")
        print(f"  示例参数: {spec.example_params}")
        tmpl_preview = spec.response_template
        if len(tmpl_preview) > 120:
            tmpl_preview = tmpl_preview[:117] + "..."
        print(f"  Tool Response Template (前120字符): {tmpl_preview}")
        print()

    attacker_tool_names = extract_attacker_tool_names_dh(attacker_cases)
    print("=== Direct-Harm 子集中的 Attacker Tools 名称 ===")
    for t in sorted(attacker_tool_names):
        print(f"- {t}")
    print(f"\n[INFO] Direct-Harm attacker tools 总数: {len(attacker_tool_names)}")
