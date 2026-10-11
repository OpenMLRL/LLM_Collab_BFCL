"""Configuration helpers for native BFCL parallel experiments."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str = "qwen"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    special_tokens: Dict[str, str] = field(default_factory=dict)
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = "sdpa"

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        *,
        require_sampling: bool = True,
    ) -> "ModelConfig":
        if require_sampling:
            missing = [
                key
                for key in ("temperature", "top_p", "top_k")
                if key not in config_dict
            ]
            if missing:
                raise ValueError(
                    f"agent_model is missing required sampling fields: {', '.join(missing)}"
                )

        def _as_optional_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid float value: {value}") from exc

        def _as_optional_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
                return None
            try:
                return int(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid int value: {value}") from exc

        temperature = _as_optional_float(config_dict.get("temperature"))
        top_p = _as_optional_float(config_dict.get("top_p"))
        top_k = _as_optional_int(config_dict.get("top_k"))
        if require_sampling and (temperature is None or top_p is None):
            raise ValueError("agent_model.temperature and agent_model.top_p must be non-null.")

        return cls(
            name=config_dict.get("name", ""),
            type=config_dict.get("type", "qwen"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            special_tokens=config_dict.get("special_tokens", {}),
            torch_dtype=(config_dict.get("torch_dtype") or config_dict.get("dtype")),
            attn_implementation=config_dict.get("attn_implementation", "sdpa"),
        )


class Config:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.data
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        return self.data.get(section, {})

    def get_agent_model_config(self) -> ModelConfig:
        model_section = self.get_section("agent_model")
        if not model_section:
            raise ValueError("No 'agent_model' section found in configuration")
        return ModelConfig.from_dict(model_section, require_sampling=True)

    def update(self, updates: Dict[str, Any]) -> None:
        self._deep_merge(self.data, updates)

    def _deep_merge(self, base: dict, updates: dict) -> None:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None) -> None:
        save_path = Path(output_path) if output_path else self.path
        with save_path.open("w", encoding="utf-8") as handle:
            yaml.dump(self.data, handle, default_flow_style=False, sort_keys=False)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--override", nargs="*", help="Override config values (format: key=value)"
    )
    return parser


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        import ast

        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_overrides(overrides: list) -> Dict[str, Any]:
    if not overrides:
        return {}

    result: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: {override}. Use key=value format."
            )
        key, value = override.split("=", 1)
        keys = key.split(".")
        current = result
        for part in keys[:-1]:
            current = current.setdefault(part, {})
        current[keys[-1]] = _parse_override_value(value)
    return result


def run_id() -> str:
    """Return a time-pid id so concurrent launches write to separate directories."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
