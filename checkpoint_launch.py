"""Optional checkpoint arguments and pre-construction W&B resume settings."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from os import PathLike, fspath


_CHECKPOINT_KEYS = {
    "resume_from_checkpoint",
    "checkpoint_dir",
    "checkpoint_interval",
    "checkpoint_total_limit",
}


def parse_checkpoint_options(config):
    """Validate only explicitly configured train arguments, without filesystem I/O."""
    section = config.get("checkpoint")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError("checkpoint must be a mapping or null.")
    unknown = set(section) - _CHECKPOINT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown checkpoint options: {', '.join(sorted(map(str, unknown)))}"
        )
    options = dict(section)
    for name in ("resume_from_checkpoint", "checkpoint_dir"):
        value = options.get(name)
        if value is not None and (
            not isinstance(value, (str, PathLike))
            or not isinstance(fspath(value), str)
            or not fspath(value).strip()
        ):
            raise ValueError(f"checkpoint.{name} must be a non-empty path or null.")
    for name in ("checkpoint_interval", "checkpoint_total_limit"):
        if name not in options:
            continue
        value = options[name]
        if name == "checkpoint_total_limit" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(
                f"checkpoint.{name} must be a positive integer"
                + (" or null." if name == "checkpoint_total_limit" else ".")
            )
    return options


def prepare_wandb_config(config, wandb_config, checkpoint_options):
    """Resolve a saved run identity before an eager trainer constructor runs."""
    section = config.get("wandb", {})
    if wandb_config is None or section is None or section is False:
        return None
    if isinstance(section, Mapping):
        enabled = section.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if (enabled is not None and not bool(enabled)) or (
            str(section.get("mode", "")).strip().lower() == "disabled"
        ):
            return None
    else:
        section = {}
    result = dict(wandb_config)
    for key in ("id", "resume"):
        if key in section:
            result[key] = section[key]
    # Key presence is deliberate: an explicit null also suppresses inference.
    if "id" in result or "resume" in result:
        return result
    resume_from = checkpoint_options.get("resume_from_checkpoint")
    if resume_from is None:
        return result

    from comlrl.utils import read_checkpoint_metadata

    identity = read_checkpoint_metadata(resume_from).get("wandb")
    if identity is not None:
        for key in ("id", "project", "entity"):
            if key in identity:
                result[key] = identity[key]
        result["resume"] = "allow"
    return result
