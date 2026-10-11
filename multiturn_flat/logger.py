"""Evaluation logging for flattened BFCL multi-turn MAGRPO runs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from multiturn_flat.rewards.multiturn_flat_reward import (
    MultiturnFlatRewardConfig,
    score_multiturn_flat_response,
)


def _prompt_key(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


def build_multiturn_flat_eval_logger(
    eval_rows: Iterable[Dict[str, Any]],
    *,
    reward_config: Dict[str, Any] | None = None,
) -> Callable[..., List[Dict[str, Any]]]:
    row_by_prompt = {
        _prompt_key(row.get("prompt") or row.get("user_prompt") or ""): row
        for row in eval_rows
    }
    cfg = MultiturnFlatRewardConfig.from_dict(reward_config or {})

    def logger(
        *,
        agent_completions_turns: List[List[List[str]]],
        prompts: List[str] | None = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        if not agent_completions_turns or prompts is None:
            return []
        metrics = []
        for sample_idx, prompt in enumerate(prompts):
            row = row_by_prompt.get(_prompt_key(prompt))
            if row is None:
                continue
            turn_completions = []
            for agent_idx in range(len(agent_completions_turns)):
                per_sample = agent_completions_turns[agent_idx][sample_idx]
                turn_completions.append(per_sample[0] if per_sample else "")
            _, detail = score_multiturn_flat_response(
                turn_completions,
                batch_item=row,
                config=cfg,
            )
            sample_metrics: Dict[str, Any] = {
                "sample_id": row.get("id", sample_idx),
                "turn_index": row.get("turn_index", ""),
            }
            for key, value in detail.items():
                if isinstance(value, (int, float)):
                    sample_metrics[f"turn_1/{key}"] = float(value)
            metrics.append(sample_metrics)
        return metrics

    return logger


def aggregate_multiturn_flat_metrics(
    metrics_list: List[Dict[str, Any]], num_turns: int = 1
) -> Dict[str, float]:
    del num_turns
    if not metrics_list:
        return {}

    aggregated: Dict[str, float] = {}
    metric_names = [
        "reward",
        "exact_match",
        "parse_success",
        "function_f1",
        "argument_match",
        "matched_calls",
        "exact_calls",
        "pred_call_count",
        "raw_call_count",
        "gold_call_count",
        "count_score",
        "balance_score",
        "overlap_count",
        "overlap_rate",
        "lazy_agents",
        "lazy_rate",
        "extra_call_rate",
        "sequence_score",
        "prefix_score",
    ]
    for metric_name in metric_names:
        key = f"turn_1/{metric_name}"
        values = [
            float(sample[key])
            for sample in metrics_list
            if key in sample and isinstance(sample[key], (int, float))
        ]
        if values:
            aggregated[key] = float(np.mean(values))

    return aggregated
