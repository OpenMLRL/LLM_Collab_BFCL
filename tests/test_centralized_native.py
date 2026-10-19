import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from comlrl.trainers.preference.collaboration import CentralizedCollaboration
from comlrl.trainers.reinforce import CentralizedMAGRPOConfig, CentralizedMAGRPOTrainer
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
    "ground_truth": [
        {"weather": {"city": ["Boston"]}},
        {"weather": {"city": ["Paris"]}},
    ],
    "annotated_answer": "SECRET_GOLD",
}
GOOD = '<agent_0> weather(city="Boston") </agent_0> <agent_1> weather(city="Paris") </agent_1>'
BAD = "<agent_0> [] </agent_0> <agent_1> [] </agent_1>"


def _runtime(role_mode="self_select"):
    return CentralizedCollaboration(
        BFCLNativeCentralizedAdapter(role_mode),
        get_native_parallel_formatters(num_agents=2, role_mode=role_mode),
        make_reward(),
        2,
    )


def test_native_prompt_states_request_and_schemas_once():
    prompt = _runtime().build_prompt(ROW)
    assert "SECRET_GOLD" not in prompt
    for once in ("Available function schemas", ROW["prompt"], '"name": "weather"'):
        assert prompt.count(once) == 1, once
    for absent in ("decentralized", "cannot communicate", "original prompt"):
        assert absent not in prompt
    assert "<agent_0>" in prompt and "<agent_1>" in prompt
    concatenated = CentralizedCollaboration(
        BFCLCentralizedComparatorAdapter(),
        get_native_parallel_formatters(num_agents=2, role_mode="self_select"),
        make_reward(),
        2,
    ).build_prompt(ROW)
    assert len(prompt) < len(concatenated)


@pytest.mark.parametrize(
    "role_mode,expected",
    [
        ("self_select", "exactly one section"),
        ("odd_even", "odd-numbered"),
        ("split_by_order", "first half"),
    ],
)
def test_native_sections_follow_the_role_split(role_mode, expected):
    assert expected in _runtime(role_mode).build_prompt(ROW)


def test_same_prompt_roles_are_rejected():
    with pytest.raises(ValueError):
        BFCLNativeCentralizedAdapter("same_prompt")


def test_native_split_gives_the_unchanged_reward():
    runtime = _runtime()
    reward = make_reward()
    for response in (GOOD, BAD, "unfinished tags"):
        outputs = runtime.split(response, ROW)
        expected = reward(*[[text] for text in outputs], batch_items=[ROW])
        assert runtime([response], batch_items=[ROW]) == expected
    assert runtime.split(GOOD, ROW) == [
        'weather(city="Boston")',
        'weather(city="Paris")',
    ]


@pytest.mark.parametrize(
    "prompt_mode,adapter_cls",
    [
        (None, BFCLCentralizedComparatorAdapter),
        ("concatenated", BFCLCentralizedComparatorAdapter),
        ("native", BFCLNativeCentralizedAdapter),
    ],
)
def test_entrypoint_selects_the_centralized_prompt(
    prompt_mode, adapter_cls, monkeypatch, tmp_path
):
    from comlrl.trainers import reinforce

    entrypoint = importlib.import_module("native_parallel.train.train_magrpo")
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(reinforce, "CentralizedMAGRPOTrainer", capture)
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
        "magrpo.collaboration_mode=centralized",
        "magrpo.parallel_training=none",
        'magrpo.agent_devices=["cpu"]',
        f"output.base_dir={tmp_path}",
        "output.save_final_model=false",
    ]
    if prompt_mode is not None:
        overrides.append(f"bfcl.centralized_prompt={prompt_mode}")
    monkeypatch.setattr(sys, "argv", ["train", "--override", *overrides])
    entrypoint.main()
    assert type(observed["centralized_adapter"]) is adapter_cls
    # The default config shares the decentralized loss and skips empty updates.
    assert observed["args"].match_decentralized_loss is True
    assert observed["args"].skip_zero_grad_steps is True


def test_entrypoint_rejects_unknown_prompt_mode(monkeypatch, tmp_path):
    entrypoint = importlib.import_module("native_parallel.train.train_magrpo")
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
        "magrpo.collaboration_mode=centralized",
        "bfcl.centralized_prompt=joint",
        f"output.base_dir={tmp_path}",
    ]
    monkeypatch.setattr(sys, "argv", ["train", "--override", *overrides])
    with pytest.raises(ValueError, match="centralized_prompt"):
        entrypoint.main()


def test_native_joint_actor_updates_with_real_bfcl_reward_and_eval():
    adapter = BFCLNativeCentralizedAdapter("self_select")
    formatters = get_native_parallel_formatters(num_agents=2, role_mode="self_select")
    reward = make_reward()
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
    trainer = CentralizedMAGRPOTrainer(
        agents=[model],
        tokenizer=tokenizer,
        num_agents=2,
        args=CentralizedMAGRPOConfig(
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
        train_dataset=[ROW],
        eval_dataset=[ROW],
        eval_logger=build_native_parallel_eval_logger([ROW]),
        eval_aggregator=aggregate_native_parallel_metrics,
    )
    trainer.verbose = False
    before = model.transformer.wte.weight.detach().clone()
    trainer.train()
    assert len(trainer.agents) == len(trainer.optimizers) == trainer.num_agents == 1
    assert trainer.env_step == 2
    assert not torch.equal(before, model.transformer.wte.weight)
    metrics = trainer.evaluate(num_eval_samples=1)
    assert metrics["eval/turn_1/exact_match"] == 1.0
