"""Shared training setup for native BFCL parallel experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import torch
from transformers import AutoTokenizer

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent

from native_parallel.config import Config, ModelConfig, run_id
from native_parallel.data import DEFAULT_NATIVE_CATEGORIES, load_native_parallel_dataset
from native_parallel.formatting import get_native_parallel_formatters
from native_parallel.logger import (
    aggregate_native_parallel_metrics,
    build_native_parallel_eval_logger,
)
from native_parallel.rewards import make_reward
from native_parallel.rewards.native_parallel_reward import (
    NativeParallelRewardConfig,
    score_native_parallel_response,
)


@dataclass
class NativeTrainComponents:
    config: Config
    model_config: ModelConfig
    model_name: str
    agent_names: Optional[List[str]]
    num_agents: int
    tokenizers: List[Any]
    train_dataset: Any
    eval_dataset: Any
    output_dir: str
    output_base_dir: str
    formatters: List[Any]
    reward_func: Any
    reward_processor: Optional[Any]
    reward_config: Dict[str, Any]
    eval_logger: Any
    eval_aggregator: Any
    metrics_callback: Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def agent_names_from_config(config: Config) -> Optional[List[str]]:
    raw = config.get("agents")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
        raise ValueError("agents must be a list of model names.")
    return [str(name) for name in raw]


def resolve_dataset_name(dataset_name: str) -> str:
    path = Path(dataset_name)
    if path.is_absolute() or path.exists():
        return str(path)
    repo_path = REPO_ROOT / dataset_name
    if repo_path.exists():
        return str(repo_path)
    return dataset_name


def optional_model_config(config: Config, section_name: str) -> Optional[ModelConfig]:
    section = config.get_section(section_name)
    if not section:
        return None
    return ModelConfig.from_dict(section, require_sampling=False)


def model_kwargs(model_config: ModelConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        kwargs["torch_dtype"] = model_config.torch_dtype
    if model_config.attn_implementation is not None:
        kwargs["attn_implementation"] = model_config.attn_implementation
    return kwargs


def magrpo_model_config(model_config: ModelConfig) -> Dict[str, Any]:
    cfg = model_kwargs(model_config)
    cfg["special_tokens"] = model_config.special_tokens
    return cfg


def actor_critic_model_config(
    actor_config: ModelConfig,
    critic_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    critic_source = critic_config or actor_config
    return {
        "model_kwargs": model_kwargs(actor_config),
        "critic_model_kwargs": model_kwargs(critic_source),
        "torch_dtype": actor_config.torch_dtype,
        "attn_implementation": actor_config.attn_implementation,
        "special_tokens": actor_config.special_tokens,
    }


def build_reward_processor(config: Config):
    if not bool_value(config.get("reward_processor.enabled", False), default=False):
        return None
    from comlrl.utils.reward_processor import RewardProcessors

    scale_factor = float(config.get("reward_processor.scale_factor", 1.0))
    reward_processor = RewardProcessors.scale(factor=scale_factor)
    shift_val = config.get("reward_processor.shift", None)
    if shift_val is None:
        return reward_processor
    shift_processor = RewardProcessors.shift(value=float(shift_val))
    return lambda value: shift_processor(reward_processor(value))


def build_trainer_args(
    args_cls: Type[Any],
    raw_section: Dict[str, Any],
    model_config: ModelConfig,
    *,
    num_agents: int,
) -> Any:
    allowed = {field.name for field in fields(args_cls)}
    kwargs = {key: value for key, value in dict(raw_section).items() if key in allowed}
    kwargs.setdefault("temperature", model_config.temperature)
    kwargs.setdefault("top_p", model_config.top_p)
    kwargs.setdefault("top_k", model_config.top_k)
    kwargs.setdefault("num_agents", num_agents)
    return args_cls(**kwargs)


def build_wandb_config(
    config: Config,
    output_dir: str,
    trainer_section: Dict[str, Any],
    *,
    algorithm_name: str,
    default_name: str,
) -> Optional[Dict[str, Any]]:
    wandb_section = config.get_section("wandb")
    if not bool_value(wandb_section.get("enabled", True), default=True):
        return None
    algo_tag = algorithm_name.lower()
    tags = wandb_section.get(
        "tags",
        [
            algo_tag,
            "bfcl",
            "v4",
            "native_parallel",
            "qwen3-4b-instruct-2507",
            "decentralized",
        ],
    )
    return {
        "project": wandb_section.get("project"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_section.get("name", default_name),
        "dir": wandb_section.get("dir", output_dir),
        "output_dir": output_dir,
        "tags": tags,
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "bfcl": config.get_section("bfcl"),
            "output": config.get_section("output"),
            "trainer": trainer_section,
        },
    }


def build_native_ac_metrics_callback(
    *,
    num_agents: int,
    reward_config: Dict[str, Any],
):
    cfg = NativeParallelRewardConfig.from_dict(reward_config or {})

    def callback(rollouts: Sequence[Any]) -> Dict[str, float]:
        grouped: Dict[tuple[int, int], Dict[str, Any]] = {}
        for sample in rollouts:
            metadata = getattr(sample, "metadata", {}) or {}
            item = metadata.get("batch_item")
            if item is None:
                continue
            generation_idx = int(metadata.get("generation_idx", 0))
            key = (id(item), generation_idx)
            group = grouped.setdefault(
                key,
                {
                    "item": item,
                    "completions": {},
                },
            )
            group["completions"][int(sample.agent_idx)] = sample.completion

        metric_rows: List[Dict[str, Any]] = []
        for group in grouped.values():
            item = group["item"]
            completions_by_agent = group["completions"]
            if any(
                agent_idx not in completions_by_agent for agent_idx in range(num_agents)
            ):
                continue
            completions = [completions_by_agent[idx] for idx in range(num_agents)]
            _, detail = score_native_parallel_response(
                completions,
                batch_item=item,
                config=cfg,
            )
            row: Dict[str, Any] = {
                "sample_id": item.get("id", ""),
            }
            for key, value in detail.items():
                if isinstance(value, (int, float)):
                    row[f"turn_1/{key}"] = float(value)
            metric_rows.append(row)

        return aggregate_native_parallel_metrics(metric_rows, num_turns=1)

    return callback


def prepare_native_components(
    config: Config,
    *,
    trainer_section_name: str,
    algorithm_name: str,
    default_output_base_dir: str,
) -> NativeTrainComponents:
    model_config = config.get_agent_model_config()
    model_name = model_config.name
    agent_names = agent_names_from_config(config)
    trainer_section = config.get_section(trainer_section_name)
    num_agents = int(trainer_section.get("num_agents", 2))

    seed_value = int(config.get("seed", 42))
    set_seed(seed_value)

    dataset_name = resolve_dataset_name(str(config.get("dataset.name")))
    train_split = str(config.get("dataset.train_split", "train"))
    eval_split = str(config.get("dataset.eval_split", "eval"))
    categories = config.get("dataset.categories", list(DEFAULT_NATIVE_CATEGORIES))
    task_types = config.get("dataset.task_types", None)

    output_base_dir = str(config.get("output.base_dir", default_output_base_dir))
    output_dir = os.path.join(output_base_dir, run_id())
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))

    train_dataset = load_native_parallel_dataset(
        dataset_name,
        split=train_split,
        categories=categories,
        task_types=task_types,
        max_samples=config.get("dataset.max_train_samples", None),
    )
    eval_dataset = load_native_parallel_dataset(
        dataset_name,
        split=eval_split,
        categories=categories,
        task_types=task_types,
        max_samples=config.get("dataset.max_eval_samples", None),
    )

    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    if agent_names:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in agent_names]
    else:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer_source)]
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tokenizer.padding_side = padding_side
        if model_config.special_tokens:
            tokenizer.add_special_tokens(model_config.special_tokens)

    role_mode = str(config.get("bfcl.role_mode", "self_select"))
    formatters = get_native_parallel_formatters(
        num_agents=num_agents,
        role_mode=role_mode,
    )
    reward_config = config.get_section("bfcl_reward")
    eval_rows = [dict(row) for row in eval_dataset]
    return NativeTrainComponents(
        config=config,
        model_config=model_config,
        model_name=model_name,
        agent_names=agent_names,
        num_agents=num_agents,
        tokenizers=tokenizers,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        output_base_dir=output_base_dir,
        formatters=formatters,
        reward_func=make_reward(reward_config),
        reward_processor=build_reward_processor(config),
        reward_config=reward_config,
        eval_logger=build_native_parallel_eval_logger(
            eval_rows,
            reward_config=reward_config,
        ),
        eval_aggregator=aggregate_native_parallel_metrics,
        metrics_callback=build_native_ac_metrics_callback(
            num_agents=num_agents,
            reward_config=reward_config,
        ),
    )


def save_final_agents_if_requested(
    components: NativeTrainComponents, trainer: Any
) -> None:
    if not bool_value(
        components.config.get("output.save_final_model", False), default=False
    ):
        return
    agents = getattr(trainer, "agents", [])
    tokenizers = getattr(trainer, "tokenizers", components.tokenizers)
    for agent_idx, agent in enumerate(agents):
        save_dir = os.path.join(components.output_dir, f"agent_{agent_idx}")
        model = getattr(agent, "model", agent)
        model.save_pretrained(save_dir)
        tokenizer = (
            tokenizers[agent_idx] if isinstance(tokenizers, list) else tokenizers
        )
        tokenizer.save_pretrained(save_dir)
