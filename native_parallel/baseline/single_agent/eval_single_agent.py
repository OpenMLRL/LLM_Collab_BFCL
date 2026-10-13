"""Evaluate a raw single-agent baseline on native BFCL parallel tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from native_parallel.config import Config, add_config_args, parse_overrides, run_id
from native_parallel.data import DEFAULT_NATIVE_CATEGORIES, load_native_parallel_dataset
from native_parallel.rewards.native_parallel_reward import (
    NativeParallelRewardConfig,
    score_native_parallel_response,
)


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


def _torch_dtype(dtype_name: Optional[str]):
    if not dtype_name:
        return None
    key = str(dtype_name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[key]


def _function_docs(example: Dict[str, Any]) -> str:
    return "\n".join(
        json.dumps(function, ensure_ascii=False, sort_keys=True)
        for function in example.get("function", [])
    )


def _build_prompt(example: Dict[str, Any]) -> str:
    user_prompt = example.get("user_prompt") or example.get("prompt") or ""
    return f"""You are a single function-calling agent.

Handle every tool-call intent you can infer from the user request.

BFCL native category: {example.get('official_category', 'unknown')}
Heuristic task type: {example.get('task_type', 'unknown')}

Available function schemas:
{_function_docs(example)}

User request:
{user_prompt}

Output requirements:
- Output all tool calls needed to satisfy the user request.
- Use Python-style function-call syntax, one call per line.
- Use keyword arguments from the function schemas.
- Do not include explanations, markdown, numbering, or reasoning.
- If no tool call is needed, output exactly: []

Example output format:
function_name(arg1="value", arg2=3)
another.function_name(flag=True)
"""


def _generation_kwargs(config: Config) -> Dict[str, Any]:
    model_config = config.get_agent_model_config()
    temperature = model_config.temperature
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(config.get("baseline.max_new_tokens", 256)),
        "do_sample": bool(temperature and float(temperature) > 0.0),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(temperature)
        if model_config.top_p is not None:
            kwargs["top_p"] = float(model_config.top_p)
        if model_config.top_k is not None:
            kwargs["top_k"] = int(model_config.top_k)
    return kwargs


def _generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    generation_kwargs: Dict[str, Any],
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    scalar_keys = [
        "reward",
        "exact_match",
        "parse_success",
        "function_f1",
        "argument_match",
        "pred_call_count",
        "raw_call_count",
        "gold_call_count",
        "balance_score",
        "overlap_rate",
        "self_duplicate_count",
        "self_duplicate_rate",
        "self_duplicate_penalty",
        "lazy_rate",
        "extra_call_rate",
    ]
    summary = {"num_samples": float(len(records))}
    for key in scalar_keys:
        summary[f"turn_1/{key}"] = _mean(
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
        )

    return summary


def _maybe_log_wandb(
    config: Config, summary: Dict[str, float], output_dir: str
) -> None:
    wandb_cfg = config.get_section("wandb")
    if not _bool(wandb_cfg.get("enabled", False), default=False):
        return
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping wandb logging.")
        return
    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity", "OpenMLRL"),
        name=wandb_cfg.get("name", "native_parallel_single_agent_baseline"),
        dir=wandb_cfg.get("dir", output_dir),
        tags=wandb_cfg.get("tags", []),
        config={
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "baseline": config.get_section("baseline"),
            "bfcl_reward": config.get_section("bfcl_reward"),
        },
    )
    wandb.log(summary)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a raw single-agent native BFCL parallel baseline."
    )
    add_config_args(parser)
    args = parser.parse_args()
    config = Config(args.config or str(DEFAULT_CONFIG))
    if args.override:
        config.update(parse_overrides(args.override))
    _set_seed(int(config.get("seed", 42)))

    model_config = config.get_agent_model_config()
    dataset_name = _resolve_dataset_name(str(config.get("dataset.name")))
    eval_dataset = load_native_parallel_dataset(
        dataset_name,
        split=str(config.get("dataset.eval_split", "eval")),
        categories=config.get("dataset.categories", list(DEFAULT_NATIVE_CATEGORIES)),
        task_types=config.get("dataset.task_types", None),
        max_samples=config.get("dataset.max_eval_samples", None),
    )

    device = str(config.get("baseline.device", "cuda:0"))
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {}
    dtype = _torch_dtype(model_config.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if model_config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_config.name, **model_kwargs).to(
        device
    )
    model.eval()

    output_base = str(config.get("baseline.output_dir", "output_single_agent_baseline"))
    output_dir = os.path.join(output_base, run_id())
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))

    generation_kwargs = _generation_kwargs(config)
    reward_cfg = NativeParallelRewardConfig.from_dict(config.get_section("bfcl_reward"))
    logging_steps = int(config.get("baseline.logging_steps", 20))
    records: List[Dict[str, Any]] = []
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(eval_dataset, start=1):
            prompt = _build_prompt(row)
            completion = _generate_one(
                model,
                tokenizer,
                prompt,
                device=device,
                generation_kwargs=generation_kwargs,
            )
            reward, detail = score_native_parallel_response(
                [completion],
                batch_item=row,
                config=reward_cfg,
            )
            record = {
                "id": row.get("id", idx - 1),
                "official_category": row.get("official_category", ""),
                "task_type": row.get("task_type", ""),
                "completion": completion,
                **detail,
            }
            record["reward"] = float(reward)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if logging_steps > 0 and idx % logging_steps == 0:
                print(f"evaluated {idx}/{len(eval_dataset)}")

    summary = _summarize(records)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _maybe_log_wandb(config, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote predictions to {predictions_path}")


if __name__ == "__main__":
    main()
