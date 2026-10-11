"""Train MAAC on native BFCL parallel function-calling tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = TASK_ROOT / "configs" / "native_parallel_maac_config.yaml"
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
if str(COMLRL_ROOT) not in sys.path:
    sys.path.insert(0, str(COMLRL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comlrl.trainers.actor_critic import MAACConfig, MAACTrainer
from native_parallel.config import Config, add_config_args, parse_overrides
from native_parallel.train.common import (
    actor_critic_model_config,
    build_trainer_args,
    build_wandb_config,
    optional_model_config,
    prepare_native_components,
    save_final_agents_if_requested,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAAC on native BFCL parallel tasks.")
    add_config_args(parser)
    args = parser.parse_args()
    if not args.config:
        args.config = str(DEFAULT_CONFIG)

    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    section_name = "maac"
    maac_section = config.get_section(section_name)
    components = prepare_native_components(
        config,
        trainer_section_name=section_name,
        algorithm_name="maac",
        default_output_base_dir="output_native_parallel_maac",
    )
    maac_args = build_trainer_args(
        MAACConfig,
        maac_section,
        components.model_config,
        num_agents=components.num_agents,
    )

    critic_config = optional_model_config(config, "critic_model")
    critic_source = critic_config.name if critic_config else components.model_name
    if not critic_source:
        raise ValueError("MAAC requires critic_model.name or agent_model.name.")

    trainer = MAACTrainer(
        agent_model=components.model_name if not components.agent_names else None,
        agents=components.agent_names,
        tokenizer=(
            components.tokenizers
            if components.agent_names
            else components.tokenizers[0]
        ),
        reward_func=components.reward_func,
        reward_processor=components.reward_processor,
        formatters=components.formatters,
        metrics_callback=components.metrics_callback,
        external_transition=None,
        args=maac_args,
        train_dataset=components.train_dataset,
        eval_dataset=components.eval_dataset,
        model_config=actor_critic_model_config(
            components.model_config,
            critic_config,
        ),
        wandb_config=build_wandb_config(
            config,
            components.output_dir,
            maac_section,
            algorithm_name="maac",
            default_name="native_parallel_maac",
        ),
        critic_model=critic_source,
    )
    trainer.train()
    save_final_agents_if_requested(components, trainer)


if __name__ == "__main__":
    main()
