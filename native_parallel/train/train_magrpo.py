"""Train MAGRPO on native BFCL parallel function-calling tasks."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = TASK_ROOT / "configs" / "native_parallel_magrpo_config.yaml"
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
if str(COMLRL_ROOT) not in sys.path:
    sys.path.insert(0, str(COMLRL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from native_parallel.config import Config, add_config_args, parse_overrides
from native_parallel.data import (
    DEFAULT_NATIVE_CATEGORIES,
    load_native_parallel_dataset,
)
from native_parallel.formatting import get_native_parallel_formatters
from native_parallel.logger import (
    aggregate_native_parallel_metrics,
    build_native_parallel_eval_logger,
)
from native_parallel.rewards import make_reward


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dataset_name(dataset_name: str) -> str:
    path = Path(dataset_name)
    if path.is_absolute() or path.exists():
        return str(path)
    repo_path = REPO_ROOT / dataset_name
    if repo_path.exists():
        return str(repo_path)
    return dataset_name


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _agent_names(config: Config) -> Optional[List[str]]:
    raw = config.get("agents")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
        raise ValueError("agents must be a list of model names.")
    return [str(name) for name in raw]


def _build_reward_processor(config: Config):
    if not _bool(config.get("reward_processor.enabled", False), default=False):
        return None
    from comlrl.utils.reward_processor import RewardProcessors

    scale_factor = float(config.get("reward_processor.scale_factor", 1.0))
    reward_processor = RewardProcessors.scale(factor=scale_factor)
    shift_val = config.get("reward_processor.shift", None)
    if shift_val is None:
        return reward_processor
    shift_processor = RewardProcessors.shift(value=float(shift_val))
    return lambda value: shift_processor(reward_processor(value))


def _build_reward(config: Config):
    return make_reward(config.get_section("bfcl_reward"))


def _wandb_config(config: Config, output_dir: str, magrpo_config: Dict[str, Any]):
    wandb_section = config.get_section("wandb")
    if not _bool(wandb_section.get("enabled", True), default=True):
        return None
    dataset_section = config.get_section("dataset")
    model_section = config.get_section("agent_model")
    bfcl_section = config.get_section("bfcl")
    tags = wandb_section.get(
        "tags",
        [
            "magrpo",
            "bfcl",
            "v4",
            "native_parallel",
            "decentralized",
        ],
    )
    return {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_section.get("name", "native_parallel_magrpo"),
        "dir": wandb_section.get("dir", output_dir),
        "tags": tags,
        "config_sections": {
            "dataset": dataset_section,
            "agent_model": model_section,
            "bfcl": bfcl_section,
            "output": config.get_section("output"),
            "trainer": magrpo_config,
        },
    }


def main() -> None:
    from comlrl.runtime import configure_job_cuda_cache

    configure_job_cuda_cache()

    parser = argparse.ArgumentParser(
        description="Train MAGRPO on native BFCL parallel tasks."
    )
    add_config_args(parser)
    args = parser.parse_args()
    if not args.config:
        args.config = str(DEFAULT_CONFIG)

    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    model_config = config.get_agent_model_config()
    model_name = model_config.name
    agent_names = _agent_names(config)
    dataset_name = _resolve_dataset_name(str(config.get("dataset.name")))
    train_split = str(config.get("dataset.train_split", "train"))
    eval_split = str(config.get("dataset.eval_split", "eval"))
    categories = config.get("dataset.categories", list(DEFAULT_NATIVE_CATEGORIES))
    task_types = config.get("dataset.task_types", None)
    seed_value = int(config.get("seed", 42))
    _set_seed(seed_value)

    output_base_dir = str(config.get("output.base_dir", "output_magrpo_bfcl"))
    job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"job_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(config, "save"):
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

    magrpo_section = config.get_section("magrpo")
    num_agents = int(magrpo_section.get("num_agents", 2))
    role_mode = str(config.get("bfcl.role_mode", "self_select"))
    reward_cfg = config.get_section("bfcl_reward")
    from comlrl.trainers.reinforce import (
        MAGRPOConfig, MAGRPOTrainer, CentralizedMAGRPOConfig, CentralizedMAGRPOTrainer,
    )

    collaboration_mode = magrpo_section.get("collaboration_mode", "decentralized")
    if collaboration_mode not in {"decentralized", "centralized"}:
        raise ValueError("magrpo.collaboration_mode must be decentralized or centralized.")
    centralized = collaboration_mode == "centralized"
    args_cls = CentralizedMAGRPOConfig if centralized else MAGRPOConfig
    trainer_cls = CentralizedMAGRPOTrainer if centralized else MAGRPOTrainer
    collaboration_kwargs = {}
    if centralized:
        from native_parallel.centralized_comparator import BFCLCentralizedComparatorAdapter

        collaboration_kwargs["centralized_adapter"] = BFCLCentralizedComparatorAdapter()

    magrpo_args = args_cls(
        num_agents=num_agents,
        num_turns=int(magrpo_section.get("num_turns", 1)),
        parallel_training=str(magrpo_section.get("parallel_training", "mp")),
        agent_devices=magrpo_section.get("agent_devices", ["cuda:0", "cuda:1"]),
        num_train_epochs=int(magrpo_section.get("num_train_epochs", 2)),
        agent_learning_rate=float(magrpo_section.get("agent_learning_rate", 2e-5)),
        logging_steps=int(magrpo_section.get("logging_steps", 20)),
        num_generations=int(magrpo_section.get("num_generations", 4)),
        max_new_tokens=int(magrpo_section.get("max_new_tokens", 256)),
        temperature=float(model_config.temperature),
        top_p=float(model_config.top_p),
        top_k=model_config.top_k,
        discount=float(magrpo_section.get("discount", 1.0)),
        joint_mode=str(magrpo_section.get("joint_mode", "cross")),
        early_termination_threshold=magrpo_section.get(
            "early_termination_threshold", None
        ),
        rollout_buffer_size=int(magrpo_section.get("rollout_buffer_size", 4)),
        train_batch_size=magrpo_section.get("train_batch_size", 4),
        advantage_normalization=_bool(
            magrpo_section.get("advantage_normalization", True), default=True
        ),
        advantage_mode=str(magrpo_section.get("advantage_mode", "mean")),
        eval_interval=int(magrpo_section.get("eval_interval", 20)),
        eval_num_samples=int(magrpo_section.get("eval_num_samples", 16)),
        eval_batch_size=int(magrpo_section.get("eval_batch_size", 1)),
        reference_kl_enabled=_bool(
            magrpo_section.get("reference_kl_enabled", False), default=False
        ),
        reference_kl_coef=float(magrpo_section.get("reference_kl_coef", 0.1)),
        reference_devices=magrpo_section.get("reference_devices", None),
    )

    formatters = get_native_parallel_formatters(
        num_agents=num_agents,
        role_mode=role_mode,
    )
    reward_func = _build_reward(config)
    eval_rows = [dict(row) for row in eval_dataset]
    eval_logger = build_native_parallel_eval_logger(
        eval_rows,
        reward_config=reward_cfg,
    )
    reward_processor = _build_reward_processor(config)

    trainer = trainer_cls(
        agent_model=model_name if not agent_names else None,
        agents=agent_names,
        num_agents=num_agents,
        tokenizer=tokenizers if agent_names else tokenizers[0],
        model_config={
            "torch_dtype": model_config.torch_dtype,
            "attn_implementation": model_config.attn_implementation,
            "special_tokens": model_config.special_tokens,
        },
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_type="bfcl",
        reward_func=reward_func,
        reward_processor=reward_processor,
        formatters=formatters,
        external_transition=None,
        wandb_config=_wandb_config(config, output_dir, magrpo_section),
        eval_logger=eval_logger,
        eval_aggregator=aggregate_native_parallel_metrics,
        args=magrpo_args,
        **collaboration_kwargs,
    )
    trainer.train()

    if _bool(config.get("output.save_final_model", False), default=False):
        for agent_idx, agent in enumerate(trainer.agents):
            save_dir = os.path.join(output_dir, f"agent_{agent_idx}")
            agent.save_pretrained(save_dir)
            trainer.tokenizers[agent_idx].save_pretrained(save_dir)


if __name__ == "__main__":
    main()
