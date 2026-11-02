"""Train MARLHF on native BFCL parallel function-calling tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = TASK_ROOT / "configs" / "native_parallel_marlhf_config.yaml"
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comlrl.trainers.pbrl import MARLHFConfig, MARLHFTrainer
from checkpoint_launch import parse_checkpoint_options, prepare_wandb_config

from native_parallel.config import Config, add_config_args, parse_overrides
from native_parallel.train.common import (
    build_centralized_adapter,
    build_trainer_args,
    build_wandb_config,
    magrpo_model_config,
    prepare_native_components,
    save_final_agents_if_requested,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MARLHF on native BFCL parallel tasks."
    )
    add_config_args(parser)
    args = parser.parse_args()
    config = Config(args.config or str(DEFAULT_CONFIG))
    if args.override:
        config.update(parse_overrides(args.override))

    checkpoint_options = parse_checkpoint_options(config)

    section_name = "marlhf"
    marlhf_section = config.get_section(section_name)
    components = prepare_native_components(
        config,
        trainer_section_name=section_name,
        algorithm_name="marlhf",
        default_output_base_dir="output_native_parallel_marlhf",
    )
    marlhf_args = build_trainer_args(
        MARLHFConfig,
        marlhf_section,
        components.model_config,
        num_agents=components.num_agents,
    )

    trainer = MARLHFTrainer(
        agent_model=components.model_name if not components.agent_names else None,
        agents=components.agent_names,
        num_agents=components.num_agents,
        tokenizer=(
            components.tokenizers
            if components.agent_names
            else components.tokenizers[0]
        ),
        model_config=magrpo_model_config(components.model_config),
        train_dataset=components.train_dataset,
        eval_dataset=components.eval_dataset,
        reward_func=components.reward_func,
        reward_processor=components.reward_processor,
        formatters=components.formatters,
        external_transition=None,
        wandb_config=prepare_wandb_config(
            config,
            build_wandb_config(
                config,
                components.output_dir,
                marlhf_section,
                algorithm_name="marlhf",
                default_name="native_parallel_marlhf",
            ),
            checkpoint_options,
        ),
        eval_logger=components.eval_logger,
        eval_aggregator=components.eval_aggregator,
        centralized_comparator_adapter=build_centralized_adapter(config),
        metrics_callback=components.metrics_callback,
        args=marlhf_args,
    )
    trainer.train(**checkpoint_options)
    save_final_agents_if_requested(components, trainer)


if __name__ == "__main__":
    main()
