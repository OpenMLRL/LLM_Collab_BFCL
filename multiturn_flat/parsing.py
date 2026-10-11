"""Parsing and canonicalization for BFCL tool-call text."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str = ""


def function_parameter_order(
    function_schemas: Sequence[Dict[str, Any]]
) -> Dict[str, List[str]]:
    order: Dict[str, List[str]] = {}
    for schema in function_schemas:
        name = str(schema.get("name", ""))
        params = schema.get("parameters", {}) or {}
        required = list(params.get("required", []) or [])
        properties = list((params.get("properties", {}) or {}).keys())
        seen = set()
        merged = []
        for key in required + properties:
            if key not in seen:
                seen.add(key)
                merged.append(key)
        order[name] = merged
    return order


def gold_to_tool_calls(ground_truth: Sequence[Dict[str, Any]]) -> List[ToolCall]:
    calls = []
    for item in ground_truth or []:
        if not isinstance(item, dict) or not item:
            continue
        name, arguments = next(iter(item.items()))
        calls.append(ToolCall(str(name), _coerce_arguments(arguments)))
    return calls


def _coerce_arguments(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return {}
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if parsed is args:
                return {}
            return _coerce_arguments(parsed)
        return {}
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    blocks = re.findall(r"```(?:json|python|text)?\s*(.*?)```", stripped, re.S | re.I)
    return "\n".join(block.strip() for block in blocks) if blocks else stripped


def _convert_json_call(obj: Any) -> List[ToolCall]:
    if isinstance(obj, list):
        calls: List[ToolCall] = []
        for item in obj:
            calls.extend(_convert_json_call(item))
        return calls
    if not isinstance(obj, dict):
        return []
    if "name" in obj:
        args = obj.get("arguments", obj.get("parameters", {}))
        return [
            ToolCall(str(obj["name"]), _coerce_arguments(args), raw=json.dumps(obj))
        ]
    if "function" in obj:
        args = obj.get("arguments", obj.get("parameters", {}))
        return [
            ToolCall(str(obj["function"]), _coerce_arguments(args), raw=json.dumps(obj))
        ]
    if len(obj) == 1:
        name, args = next(iter(obj.items()))
        if isinstance(args, dict):
            return [ToolCall(str(name), dict(args), raw=json.dumps(obj))]
    return []


def _try_parse_json(text: str) -> List[ToolCall]:
    candidates = [text.strip()]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return _convert_json_call(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return []


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        return -value if isinstance(value, (int, float)) else value
    return ast.unparse(node) if hasattr(ast, "unparse") else None


def _normalize_python_literals(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text, flags=re.I)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.I)
    text = re.sub(r"\bnull\b", "None", text, flags=re.I)
    return text


def _call_from_ast(
    node: ast.Call,
    *,
    parameter_order: Dict[str, List[str]],
    raw: str,
) -> Optional[ToolCall]:
    name = _call_name(node.func)
    if not name:
        return None
    args: Dict[str, Any] = {}
    ordered_params = parameter_order.get(name, [])
    for idx, value_node in enumerate(node.args):
        if idx < len(ordered_params):
            args[ordered_params[idx]] = _literal(value_node)
    for kw in node.keywords:
        if kw.arg is not None:
            args[kw.arg] = _literal(kw.value)
    return ToolCall(name=name, arguments=args, raw=raw.strip())


def _extract_call_spans(text: str, allowed_names: Sequence[str]) -> List[str]:
    snippets: List[str] = []
    names = sorted(set(allowed_names), key=len, reverse=True)
    for name in names:
        pattern = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\(")
        for match in pattern.finditer(text):
            start = match.start()
            depth = 0
            end = None
            for idx in range(match.end() - 1, len(text)):
                char = text[idx]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end is not None:
                snippets.append(text[start:end])
    return snippets


def _try_parse_ast(
    text: str,
    *,
    function_schemas: Sequence[Dict[str, Any]],
) -> List[ToolCall]:
    parameter_order = function_parameter_order(function_schemas)
    allowed_names = list(parameter_order.keys())
    snippets = []
    stripped = text.strip()
    if stripped:
        snippets.append(stripped)
        if not stripped.startswith("["):
            snippets.append(f"[{stripped}]")
    snippets.extend(_extract_call_spans(text, allowed_names))

    calls: List[ToolCall] = []
    for snippet in snippets:
        candidate = _normalize_python_literals(snippet)
        try:
            tree = ast.parse(candidate, mode="eval")
        except SyntaxError:
            try:
                tree = ast.parse(candidate, mode="exec")
            except SyntaxError:
                continue

        nodes: List[ast.Call] = []
        if isinstance(tree, ast.Expression):
            body = tree.body
            if isinstance(body, ast.Call):
                nodes = [body]
            elif isinstance(body, (ast.List, ast.Tuple)):
                nodes = [elt for elt in body.elts if isinstance(elt, ast.Call)]
        else:
            for item in tree.body:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                    nodes.append(item.value)

        for node in nodes:
            call = _call_from_ast(node, parameter_order=parameter_order, raw=snippet)
            if call is not None:
                calls.append(call)
        if calls:
            break
    return calls


def _normal_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    text = str(value).strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def canonical_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(canonical_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(canonical_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), canonical_value(val)) for key, val in value.items())
        )
    return _normal_scalar(value)


def canonical_call_key(call: ToolCall) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    return (
        call.name,
        tuple(
            sorted(
                (str(key), canonical_value(value))
                for key, value in call.arguments.items()
            )
        ),
    )


def dedupe_calls(calls: Iterable[ToolCall]) -> List[ToolCall]:
    seen = set()
    unique = []
    for call in calls:
        key = canonical_call_key(call)
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def parse_tool_calls(
    text: str,
    *,
    function_schemas: Sequence[Dict[str, Any]],
) -> List[ToolCall]:
    """Parse model output into tool calls.

    Supports JSON call objects and Python-style BFCL call syntax.
    """
    if not text or text.strip() == "[]":
        return []
    cleaned = _strip_fences(text)
    calls = _try_parse_json(cleaned)
    if calls:
        return calls
    return _try_parse_ast(cleaned, function_schemas=function_schemas)
