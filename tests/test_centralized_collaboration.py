import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from comlrl.trainers.centralized import CentralizedCollaboration
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
from native_parallel.centralized_comparator import BFCLCentralizedComparatorAdapter
from native_parallel.centralized_native import BFCLNativeCentralizedAdapter
from native_parallel.formatting import get_native_parallel_formatters
from native_parallel.logger import (
    aggregate_native_parallel_metrics,
    build_native_parallel_eval_logger,
)
from native_parallel.rewards import make_reward


ROW = {
    "id": "two-cities",
    "prompt": "Get the weather in Boston and Paris.",
    "function": [{"name": "weather", "parameters": {"city": "string"}}],
}


@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_magrpo_entrypoint_selects_joint_actor(mode, monkeypatch, tmp_path):
    from comlrl.trainers import reinforce

    entrypoint = importlib.import_module("native_parallel.train.train_magrpo")
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(reinforce, "MAGRPOTrainer", capture)
    monkeypatch.setattr(
        entrypoint,
        "load_native_parallel_dataset",
        lambda *a, **kw: [{"prompt": "Task", "id": "task"}],
    )
    monkeypatch.setattr(
        entrypoint.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(pad_token="pad", eos_token="eos"),
    )
    overrides = [
        f"magrpo.collaboration_mode={mode}",
        f"output.base_dir={tmp_path}",
        "output.save_final_model=false",
    ]
    if mode == "centralized":
        overrides += ["magrpo.parallel_training=none", 'magrpo.agent_devices=["cpu"]']
    monkeypatch.setattr(sys, "argv", ["train", "--override", *overrides])
    entrypoint.main()
    assert observed["num_agents"] == 2
    assert ("centralized_adapter" in observed) == (mode == "centralized")
    assert getattr(observed["args"], "collaboration_mode", "decentralized") == mode
    if mode == "decentralized":
        assert observed["args"].agent_devices == ["cuda:0", "cuda:1"]
        assert observed["args"].parallel_training == "mp"


def _run_preference_entrypoint(section, trainer_name, overrides, monkeypatch):
    """Run a pbrl launcher on stub components and return the trainer's kwargs."""
    entrypoint = importlib.import_module(f"native_parallel.train.train_{section}")
    config = entrypoint.Config(str(entrypoint.DEFAULT_CONFIG))
    assert config.get(f"{section}.collaboration_mode") in (None, "decentralized")
    components = SimpleNamespace(
        model_name="local-test-model",
        agent_names=None,
        num_agents=2,
        model_config=config.get_agent_model_config(),
        tokenizers=[object()],
        train_dataset=[],
        eval_dataset=[],
        reward_func=lambda left, right: [1.0],
        reward_processor=None,
        formatters=[lambda _: "left tools", lambda _: "right tools"],
        output_dir="unused",
        eval_logger=None,
        eval_aggregator=None,
        metrics_callback=None,
    )
    observed = {}

    def create_trainer(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(
        entrypoint, "prepare_native_components", lambda *a, **kw: components
    )
    monkeypatch.setattr(entrypoint, "build_wandb_config", lambda *a, **kw: {})
    monkeypatch.setattr(entrypoint, "save_final_agents_if_requested", lambda *a: None)
    monkeypatch.setattr(entrypoint, trainer_name, create_trainer)
    monkeypatch.setattr(sys, "argv", ["train", "--override", *overrides])
    entrypoint.main()
    return observed


@pytest.mark.parametrize(
    "section,trainer_name",
    [
        ("madpo", "MADPOTrainer"),
        ("marlhf", "MARLHFTrainer"),
        ("madpo_iter", "MADPOIterTrainer"),
        ("marlhf_iter", "MARLHFIterTrainer"),
    ],
)
@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_entrypoint_passes_bfcl_adapter(section, trainer_name, mode, monkeypatch):
    overrides = [f"{section}.collaboration_mode={mode}"]
    if mode == "centralized":
        overrides += [
            f"{section}.parallel_training=none",
            f'{section}.agent_devices=["cpu"]',
        ]
    observed = _run_preference_entrypoint(section, trainer_name, overrides, monkeypatch)
    assert observed["num_agents"] == 2
    assert observed["args"].collaboration_mode == mode
    if mode == "centralized":
        runtime = CentralizedCollaboration(
            observed["centralized_comparator_adapter"],
            observed["formatters"],
            observed["reward_func"],
            2,
        )
        assert "function-calling agents" in runtime.build_prompt({})
        response = '<agent_0>weather(city="Boston")</agent_0><agent_1>[]</agent_1>'
        assert runtime.split(response, {}) == ['weather(city="Boston")', "[]"]
        if section.endswith("_iter"):
            assert observed["args"].comparator_generation_mode == "centralized"


@pytest.mark.parametrize("prompt_mode", [None, "concatenated", "native"])
@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_maac_entrypoint_selects_the_joint_actor_prompt(mode, prompt_mode, monkeypatch):
    overrides = [f"maac.collaboration_mode={mode}"]
    if mode == "centralized":
        overrides += ["maac.parallel_training=none", 'maac.agent_devices=["cpu"]']
    if prompt_mode is not None:
        overrides.append(f"bfcl.centralized_prompt={prompt_mode}")
    observed = _run_preference_entrypoint("maac", "MAACTrainer", overrides, monkeypatch)
    assert observed["args"].collaboration_mode == mode
    adapter = observed.get("centralized_adapter")
    if mode == "decentralized":
        assert adapter is None
    elif prompt_mode == "native":
        assert type(adapter) is BFCLNativeCentralizedAdapter
    else:
        assert type(adapter) is BFCLCentralizedComparatorAdapter


@pytest.mark.parametrize(
    "section,trainer_name",
    [("madpo_iter", "MADPOIterTrainer"), ("marlhf_iter", "MARLHFIterTrainer")],
)
@pytest.mark.parametrize("prompt_mode", [None, "native"])
def test_centralized_comparator_follows_the_centralized_prompt(
    section, trainer_name, prompt_mode, monkeypatch
):
    # Dec-Cen: decentralized agents are compared with one centralized comparator.
    overrides = [
        f"{section}.comparator_generation_mode=centralized",
        "bfcl.role_mode=odd_even",
    ]
    if prompt_mode is not None:
        overrides.append(f"bfcl.centralized_prompt={prompt_mode}")
    observed = _run_preference_entrypoint(section, trainer_name, overrides, monkeypatch)
    assert observed["args"].collaboration_mode == "decentralized"
    assert observed["args"].comparator_generation_mode == "centralized"
    adapter = observed["centralized_comparator_adapter"]
    formatters = get_native_parallel_formatters(num_agents=2, role_mode="odd_even")
    roles = [format_role(ROW) for format_role in formatters]
    prompt = adapter.build_prompt(ROW, roles)
    if prompt_mode == "native":
        assert type(adapter) is BFCLNativeCentralizedAdapter
        assert adapter.role_mode == "odd_even"
        assert prompt.count(ROW["prompt"]) == 1
        assert not any(role in prompt for role in roles)
    else:
        assert type(adapter) is BFCLCentralizedComparatorAdapter
        assert all(role in prompt for role in roles)


GOOD = '<agent_0> weather(city="Boston") </agent_0> <agent_1> weather(city="Paris") </agent_1>'
BAD = "<agent_0> [] </agent_0> <agent_1> [] </agent_1>"


def test_joint_actor_updates_with_real_bfcl_reward_and_eval():
    row = {
        "id": "two-cities",
        "prompt": "Get the weather in Boston and Paris.",
        "function": [{"name": "weather", "parameters": {"city": "string"}}],
        "ground_truth": [
            {"weather": {"city": ["Boston"]}},
            {"weather": {"city": ["Paris"]}},
        ],
        "annotated_answer": "SECRET_GOLD",
    }
    adapter = BFCLCentralizedComparatorAdapter()
    formatters = get_native_parallel_formatters(
        num_agents=2, role_mode="split_by_order"
    )
    reward = make_reward()
    runtime = CentralizedCollaboration(adapter, formatters, reward, 2)
    prompt = runtime.build_prompt(row)
    assert "SECRET_GOLD" not in prompt
    assert "first half" in prompt and "second half" in prompt
    assert "Available function schemas" in prompt
    for response in (GOOD, BAD, "unfinished tags"):
        outputs = runtime.split(response, row)
        expected = reward(*[[text] for text in outputs], batch_items=[row])
        assert runtime([response], batch_items=[row]) == expected

    vocabulary = ["[UNK]", "[PAD]", "[EOS]", *dict.fromkeys((GOOD + " " + BAD).split())]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_embd=8,
            n_head=1,
            n_layer=1,
            n_positions=2048,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )

    def generate(*, input_ids, num_return_sequences, **kwargs):
        sequences = [
            torch.cat(
                [
                    input_ids[0],
                    torch.tensor(tokenizer.encode(GOOD if i % 2 == 0 else BAD)),
                ]
            )
            for i in range(num_return_sequences)
        ]
        return SimpleNamespace(sequences=torch.stack(sequences))

    model.generate = generate
    trainer = MAGRPOTrainer(
        agents=[model],
        tokenizer=tokenizer,
        num_agents=2,
        args=MAGRPOConfig(
            collaboration_mode="centralized",
            num_turns=1,
            agent_devices=["cpu"],
            parallel_training="none",
            num_generations=2,
            num_train_epochs=1,
            eval_interval=0,
            max_new_tokens=8,
            rollout_buffer_size=1,
            train_batch_size=1,
            agent_learning_rate=1e-3,
        ),
        centralized_adapter=adapter,
        formatters=formatters,
        reward_func=reward,
        train_dataset=[row],
        eval_dataset=[row],
        eval_logger=build_native_parallel_eval_logger([row]),
        eval_aggregator=aggregate_native_parallel_metrics,
    )
    trainer.verbose = False
    before = model.transformer.wte.weight.detach().clone()
    trainer.train()
    assert len(trainer.agents) == len(trainer.optimizers) == trainer.num_agents == 1
    assert trainer.args.num_agents == 2 and trainer.env_step == 2
    assert not torch.equal(before, model.transformer.wte.weight)
    metrics = trainer.evaluate(num_eval_samples=1)
    assert metrics["eval/turn_1/exact_match"] == 1.0
    assert metrics["eval/turn_1/pred_call_count"] == 2.0
