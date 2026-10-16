"""Launchers forward optional checkpoints before eager trainer logging starts."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import comlrl.utils
from checkpoint_launch import parse_checkpoint_options, prepare_wandb_config


class Config:
    def __init__(self, **data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def get_section(self, key):
        value = self.data.get(key, {})
        return dict(value) if isinstance(value, dict) else {}


def forbid_metadata(monkeypatch):
    monkeypatch.setattr(
        comlrl.utils,
        "read_checkpoint_metadata",
        lambda *args: pytest.fail("unexpected checkpoint metadata I/O"),
    )


@pytest.mark.parametrize("section", [None, {}])
def test_unconfigured_checkpoint_has_no_defaults_or_metadata_io(section, monkeypatch):
    forbid_metadata(monkeypatch)
    options = parse_checkpoint_options(Config(checkpoint=section))
    assert options == {}
    original = {"project": "current", "name": "unchanged"}
    assert prepare_wandb_config(Config(), original, options) == original
    assert parse_checkpoint_options(Config()) == {}


def test_options_preserve_explicit_values_without_creating_directories(tmp_path):
    values = {
        "resume_from_checkpoint": "~/saved/checkpoint-2",
        "checkpoint_dir": tmp_path / "snapshots",
        "checkpoint_interval": 2,
        "checkpoint_total_limit": None,
    }
    assert parse_checkpoint_options(Config(checkpoint=values)) == values
    assert not (tmp_path / "snapshots").exists()


@pytest.mark.parametrize(
    "section",
    [
        [],
        "checkpoint",
        False,
        {"unknown": 1},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": True},
        {"checkpoint_interval": 1.5},
        {"checkpoint_interval": "2"},
        {"checkpoint_interval": None},
        {"checkpoint_total_limit": -1},
        {"checkpoint_total_limit": False},
        {"checkpoint_total_limit": 2.5},
        {"checkpoint_dir": " "},
        {"checkpoint_dir": 3},
        {"resume_from_checkpoint": False},
        {"resume_from_checkpoint": ""},
    ],
)
def test_invalid_options_rejected(section):
    with pytest.raises(ValueError, match="checkpoint"):
        parse_checkpoint_options(Config(checkpoint=section))


@pytest.mark.parametrize(
    "explicit", [{"id": "chosen"}, {"resume": "never"}, {"id": None}, {"resume": None}]
)
def test_explicit_wandb_choice_skips_metadata_and_is_forwarded(explicit, monkeypatch):
    forbid_metadata(monkeypatch)
    original = {"project": "current", "name": "keep"}
    actual = prepare_wandb_config(
        Config(wandb=explicit), original, {"resume_from_checkpoint": "not-opened"}
    )
    assert actual == {**original, **explicit}
    assert original == {"project": "current", "name": "keep"}


@pytest.mark.parametrize(
    "section",
    [None, False, {"enabled": False}, {"enabled": "false"}, {"mode": "disabled"}],
)
@pytest.mark.parametrize("built", [None, {"project": "builder-default"}])
def test_disabled_wandb_skips_checkpoint_metadata(section, built, monkeypatch):
    forbid_metadata(monkeypatch)
    assert (
        prepare_wandb_config(
            Config(wandb=section),
            built,
            {"resume_from_checkpoint": "not-opened"},
        )
        is None
    )


@pytest.mark.parametrize(
    "identity", [None, {"id": "saved", "project": "old-project", "entity": "old-team"}]
)
def test_saved_identity_has_priority_and_legacy_preserves_logging(
    identity, monkeypatch
):
    seen = []

    def read(path):
        seen.append(path)
        return {"wandb": identity}

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", read)
    original = {
        "project": "new-project",
        "entity": "new-team",
        "name": "keep",
        "tags": ["bfcl"],
        "dir": "logs",
        "output_dir": "outputs",
        "config_sections": {"trainer": {"num_agents": 2}},
    }
    actual = prepare_wandb_config(
        Config(wandb={"project": "new-project"}),
        original,
        {"resume_from_checkpoint": "saved-checkpoint"},
    )
    assert seen == ["saved-checkpoint"]
    expected = (
        original if identity is None else {**original, **identity, "resume": "allow"}
    )
    assert actual == expected
    assert original["project"] == "new-project" and "id" not in original


ENTRIES = [
    ("native_parallel.train.train_magrpo", "MAGRPOTrainer"),
    ("native_parallel.train.train_iac", "IACTrainer"),
    ("native_parallel.train.train_maac", "MAACTrainer"),
    ("native_parallel.train.train_madpo", "MADPOTrainer"),
    ("native_parallel.train.train_marlhf", "MARLHFTrainer"),
    ("native_parallel.train.train_madpo_iter", "MADPOIterTrainer"),
    ("native_parallel.train.train_marlhf_iter", "MARLHFIterTrainer"),
    ("multiturn_flat.train.train_magrpo", "MAGRPOTrainer"),
]


def stub_launcher(monkeypatch, module_name, trainer_name, events):
    entry = importlib.import_module(module_name)

    def create(**kwargs):
        events.append(("construct", kwargs))
        return SimpleNamespace(
            train=lambda **options: events.append(("train", options))
        )

    if trainer_name == "MAGRPOTrainer":
        from comlrl.trainers import reinforce

        monkeypatch.setattr(reinforce, trainer_name, create)
        loader = (
            "load_native_parallel_dataset"
            if module_name.startswith("native")
            else "load_multiturn_flat_dataset"
        )
        monkeypatch.setattr(entry, loader, lambda *args, **kwargs: [])
        monkeypatch.setattr(entry, "_set_seed", lambda seed: None)
        monkeypatch.setattr(
            entry.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(pad_token="pad", eos_token="eos"),
        )
    else:
        config = entry.Config(str(entry.DEFAULT_CONFIG))
        components = SimpleNamespace(
            model_name="local-test-model",
            agent_names=None,
            num_agents=2,
            model_config=config.get_agent_model_config(),
            tokenizers=[object()],
            train_dataset=[],
            eval_dataset=[],
            reward_func=lambda *args: [1.0],
            reward_processor=None,
            formatters=[lambda row: "left", lambda row: "right"],
            output_dir="unused",
            eval_logger=None,
            eval_aggregator=None,
            metrics_callback=None,
        )
        monkeypatch.setattr(
            entry, "prepare_native_components", lambda *args, **kwargs: components
        )
        monkeypatch.setattr(entry, "save_final_agents_if_requested", lambda *args: None)
        monkeypatch.setattr(entry, trainer_name, create)
    return entry


@pytest.mark.parametrize("module_name,trainer_name", ENTRIES)
@pytest.mark.parametrize(
    "mode",
    [
        "default",
        "resume",
        "explicit",
        "disabled",
        "legacy",
        "disabled_null",
        "disabled_false",
        "disabled_string",
        "disabled_mode",
    ],
)
def test_all_launchers_resolve_logging_before_construction_and_forward_options(
    module_name, trainer_name, mode, monkeypatch, tmp_path
):
    events = []
    entry = stub_launcher(monkeypatch, module_name, trainer_name, events)

    def metadata(path):
        assert mode in {"resume", "legacy"}, "unexpected metadata read"
        events.append(("metadata", path))
        return {
            "wandb": (
                None
                if mode == "legacy"
                else {
                    "id": "saved-run",
                    "project": "saved-project",
                    "entity": "saved-team",
                }
            )
        }

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", metadata)
    overrides = [f"output.base_dir={tmp_path}", "output.save_final_model=false"]
    expected = {}
    if mode != "default":
        expected = {
            "resume_from_checkpoint": "saved-checkpoint",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval": 2,
            "checkpoint_total_limit": 3,
        }
        overrides.extend(f"checkpoint.{key}={value}" for key, value in expected.items())
    if mode == "explicit":
        overrides.append("wandb.resume=never")
    disabled_override = {
        "disabled": "wandb.enabled=false",
        "disabled_null": "wandb=null",
        "disabled_false": "wandb=false",
        "disabled_string": "wandb.enabled='false'",
        "disabled_mode": "wandb.mode=disabled",
    }
    if mode in disabled_override:
        overrides.append(disabled_override[mode])
    monkeypatch.setattr(sys, "argv", ["train", "--override", *overrides])
    entry.main()
    assert [name for name, value in events] == (
        ["metadata", "construct", "train"]
        if mode in {"resume", "legacy"}
        else ["construct", "train"]
    )
    assert events[-1] == ("train", expected)
    logging = events[-2][1]["wandb_config"]
    if mode.startswith("disabled"):
        assert logging is None
    elif mode == "resume":
        assert {key: logging[key] for key in ("id", "project", "entity", "resume")} == {
            "id": "saved-run",
            "project": "saved-project",
            "entity": "saved-team",
            "resume": "allow",
        }
        assert "tags" in logging and "config_sections" in logging and "dir" in logging
    elif mode == "explicit":
        assert logging["resume"] == "never" and "id" not in logging
    else:
        assert "id" not in logging and "resume" not in logging
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("module_name,trainer_name", ENTRIES)
def test_invalid_options_fail_before_components_or_trainer(
    module_name, trainer_name, monkeypatch
):
    entry = importlib.import_module(module_name)
    if hasattr(entry, "prepare_native_components"):
        monkeypatch.setattr(
            entry,
            "prepare_native_components",
            lambda *args, **kwargs: pytest.fail("components constructed"),
        )
    else:
        monkeypatch.setattr(
            entry, "_set_seed", lambda seed: pytest.fail("startup continued")
        )
    monkeypatch.setattr(
        sys, "argv", ["train", "--override", "checkpoint.checkpoint_interval=0"]
    )
    with pytest.raises(ValueError, match="checkpoint.checkpoint_interval"):
        entry.main()
