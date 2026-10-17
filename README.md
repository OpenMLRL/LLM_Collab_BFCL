# LLM Collaboration – BFCL v4

Function-calling environments for [CoMLRL](https://github.com/OpenMLRL/CoMLRL), built from official BFCL v4 data. Two agents split the tool calls for a request and are trained with a shared reward. Each task directory (`native_parallel/`, `multiturn_flat/`) is self-contained, with its own configs, data loader, formatter, parser, logger, reward, and entrypoints.

## Installation

CoMLRL can be installed from PyPI or conda-forge (`conda install -c conda-forge comlrl`). Users should also install a PyTorch build compatible with their device.

```bash
pip install comlrl
git clone https://github.com/OpenMLRL/LLM_Collab_BFCL.git
cd LLM_Collab_BFCL
```

A `CoMLRL/` checkout next to this repository takes precedence over the installed package, so local CoMLRL changes are used without reinstalling.

## Quick Start

The following command trains two agents with MAGRPO on native parallel function calling, using the default config:

```bash
python native_parallel/train/train_magrpo.py
```

Any configuration value can be changed with `--override`, for example to filter by task type or to give the agents fixed first-half and second-half roles:

```bash
python native_parallel/train/train_magrpo.py \
  --override dataset.task_types='["travel/local_services/logistics"]' bfcl.role_mode=split_by_order
```

## Tasks

| Task | Dataset | Categories | Reward |
| --- | --- | --- | --- |
| `native_parallel/` | `OpenMLRL/BFCL-V4-Parallel-Native` | `parallel`, `parallel_multiple` (live categories through `dataset.categories`) | Merged calls compared as an unordered set with the gold calls |
| `multiturn_flat/` | `OpenMLRL/BFCL-V4-Parallel-Multi-Turn` | `multi_turn_base_step`, `multi_turn_long_context_step`, `multi_turn_miss_func_step`, `multi_turn_miss_param_step` | Calls in agent order compared with the current turn's ordered gold calls |

Each `multiturn_flat` row is one turn of an original BFCL multi-turn trajectory, with the earlier context in `user_prompt`; it is not a stateful environment with tool execution. Rows provide `official_category`, `task_type` (an OpenMLRL keyword category, not an official BFCL label), `user_prompt`, `function`, `ground_truth`, and, for multi-turn steps, `turn_index`. Default configs use the first half of each split, such as `train[:160]` and `eval[:40]`; setting `dataset.train_split` and `dataset.eval_split` to `train` and `eval` uses the full splits.

Both rewards check parsing, function names, argument values, and call counts, add an exact-match bonus, and penalize duplicated calls across agents and agents that emit no calls. The native reward also rewards an even split of calls between the agents; the multi-turn reward adds sequence, prefix, and extra-call terms.

## Training Scripts

| Script | Method |
| --- | --- |
| `native_parallel/train/train_magrpo.py` | MAGRPO |
| `native_parallel/train/train_iac.py`, `train_maac.py` | IAC, MAAC |
| `native_parallel/train/train_madpo.py`, `train_madpo_iter.py` | MADPO and its iterative variant |
| `native_parallel/train/train_marlhf.py`, `train_marlhf_iter.py` | MARLHF and its iterative variant |
| `multiturn_flat/train/train_magrpo.py` | MAGRPO |
| `<task>/baseline/single_agent/eval_single_agent.py` | Single-agent baseline without training |

Defaults use two Qwen3-4B-Instruct-2507 agents with `self_select` role prompting. The native defaults are set to about 2,560 logged environment steps, or 5,120 for the iterative preference methods. Preference methods use `joint_mode: aligned`, and MAAC's critic and MARLHF's reward model default to a third GPU. The baselines let one agent answer the evaluation split and report `turn_1/exact_match`, writing `predictions.jsonl` and `summary.json` to their output directory.

## Options

**Centralized collaboration.** For native parallel tasks, one model can generate both roles. Because the default config places the agents on two devices, the device settings have to change as well:

```bash
python native_parallel/train/train_magrpo.py --override \
  magrpo.collaboration_mode=centralized magrpo.num_turns=1 \
  magrpo.parallel_training=none 'magrpo.agent_devices=["cuda:0"]'
```

The same settings apply to the other methods with their own prefix, such as `madpo_iter.`. `num_agents` stays 2 for the two roles, while only one model is loaded. The joint response contains `<agent_0>` and `<agent_1>` sections, which are split only for rewards and evaluation. `max_new_tokens` then limits the whole joint response, and MARLHF's `reward_max_length` has to fit the joint prompt and both responses.

**Native centralized prompt (MAGRPO).** With `self_select`, both agent prompts are identical, so the concatenated centralized prompt repeats the function schemas, the user request and the "you cannot communicate" instructions. `bfcl.centralized_prompt=native` states the request and schemas once and asks one planner for both `<agent_0>` / `<agent_1>` sections, split the same way as the configured `role_mode`; parsing and reward are unchanged, and the default stays `concatenated`. With a 512-token joint budget on one B200, the concatenated prompt runs out of memory with `eager` attention on the longest rows and peaks at about 99 GB with `sdpa`, so centralized runs, and the decentralized runs they are compared with, use `agent_model.attn_implementation=sdpa`:

```bash
python native_parallel/train/train_magrpo.py --override \
  magrpo.collaboration_mode=centralized magrpo.num_turns=1 \
  magrpo.parallel_training=none 'magrpo.agent_devices=["cuda:0"]' \
  magrpo.max_new_tokens=512 bfcl.centralized_prompt=native \
  agent_model.attn_implementation=sdpa
```

**Checkpoints.** A top-level `checkpoint` section is passed to CoMLRL's `trainer.train()`; without it, no training checkpoints are saved.

```yaml
checkpoint:
  checkpoint_dir: checkpoints/native_parallel
  checkpoint_interval: 1
  checkpoint_total_limit: 3
  # resume_from_checkpoint: checkpoints/native_parallel/checkpoint-2
```

Resuming does not save new checkpoints unless `checkpoint_dir` is also set. A resumed run continues logging to the saved W&B run unless `wandb.id` or `wandb.resume` is set.
