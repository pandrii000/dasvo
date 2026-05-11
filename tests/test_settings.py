from dasvo.settings import (
    BACKEND_ASSET_ROOTS,
    CONFIG_FILE,
    DATA_ROOT,
    DEGRADATIONS,
    FRONTEND_ASSET_ROOTS,
    PAPER_ASSETS_ROOT,
    PAPER_FIGURES_ROOT,
    PAPER_TABLES_ROOT,
    PROJECT_ROOT,
    SETTINGS,
    DasvoSettings,
    load_settings,
)
from pydantic import ValidationError
import pytest
import yaml


def test_default_config_file_is_loaded():
    assert CONFIG_FILE == PROJECT_ROOT / "configs" / "default.yaml"
    assert CONFIG_FILE.is_file()
    assert SETTINGS.dataset.sequences_file == PROJECT_ROOT / "tartan_test.txt"
    assert DATA_ROOT == PROJECT_ROOT / "assets" / "data" / "tartanair"


def test_default_yaml_contains_full_schema_sections():
    raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    assert set(raw) == {
        "paths",
        "dataset",
        "camera",
        "download",
        "frontends",
        "backends",
        "degradations",
        "precomputed",
        "experiment",
        "evaluation",
        "publication",
    }
    DasvoSettings.model_validate({"config_file": CONFIG_FILE, **raw})


def test_paper_asset_paths_point_to_active_manuscript_tree():
    assert PAPER_ASSETS_ROOT.name == "depth-assisted-sparse-visual-odometry"
    assert PAPER_FIGURES_ROOT == PAPER_ASSETS_ROOT / "figures"
    assert PAPER_TABLES_ROOT == PAPER_ASSETS_ROOT / "tables"


def test_precomputed_asset_roots_cover_frontend_and_backend_methods():
    assert FRONTEND_ASSET_ROOTS["orb"] == PROJECT_ROOT / "assets" / "interim" / "frontend" / "orb"
    assert FRONTEND_ASSET_ROOTS["klt"] == PROJECT_ROOT / "assets" / "interim" / "frontend" / "klt"
    assert BACKEND_ASSET_ROOTS["essential"] == PROJECT_ROOT / "assets" / "interim" / "backend" / "essential"
    assert BACKEND_ASSET_ROOTS["pnp"] == PROJECT_ROOT / "assets" / "interim" / "backend" / "pnp"


def test_degradation_presets_are_explicit_and_parameterized():
    assert set(DEGRADATIONS) == {
        "none",
        "blur_mild",
        "blur_heavy",
        "gaussian_noise",
        "jpeg_low",
        "low_light",
    }
    assert SETTINGS.degradations.presets["blur_heavy"].parameters["kernel_size"] == 11
    assert SETTINGS.degradations.presets["gaussian_noise"].parameters["std"] == 12.0
    assert SETTINGS.degradations.presets["jpeg_low"].parameters["quality"] == 25


def test_load_settings_accepts_full_custom_yaml(tmp_path):
    config = tmp_path / "experiment.yaml"
    raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    raw["paths"]["data_root"] = "assets/data/custom"
    raw["dataset"]["sequences_file"] = "tartan_test.txt"
    raw["download"]["source"] = "cmu"
    raw["precomputed"]["frontend_roots"]["orb"] = "assets/interim/custom_frontend/orb"
    raw["experiment"]["frame_strides"] = [1, 3]
    raw["experiment"]["degradations"] = ["none", "blur_mild"]
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    settings = load_settings(config, environ={})

    assert settings.paths.data_root == PROJECT_ROOT / "assets" / "data" / "custom"
    assert settings.dataset.sequences_file == PROJECT_ROOT / "tartan_test.txt"
    assert settings.download.source == "s3"
    assert settings.precomputed.frontend_roots["orb"] == (
        PROJECT_ROOT / "assets" / "interim" / "custom_frontend" / "orb"
    )
    assert settings.experiment.frame_strides == (1, 3)
    assert settings.experiment.degradations == ("none", "blur_mild")


def test_load_settings_applies_environment_overrides(tmp_path):
    env_data = tmp_path / "data"
    env_sequences = tmp_path / "seqs.txt"

    settings = load_settings(
        environ={
            "DASVO_DATA_ROOT": str(env_data),
            "DASVO_SEQUENCES_FILE": str(env_sequences),
            "DASVO_DOWNLOAD_SOURCE": "s3",
            "DASVO_FRONTEND_ASSETS_ROOT": str(tmp_path / "frontend"),
            "DASVO_BACKEND_ASSETS_ROOT": str(tmp_path / "backend"),
            "DASVO_NUM_WORKERS": "4",
        }
    )

    assert settings.paths.data_root == env_data.resolve()
    assert settings.dataset.sequences_file == env_sequences.resolve()
    assert settings.download.source == "s3"
    assert settings.precomputed.frontend_root == (tmp_path / "frontend").resolve()
    assert settings.precomputed.backend_root == (tmp_path / "backend").resolve()
    assert settings.experiment.num_workers == 4


def test_pydantic_rejects_unknown_config_keys(tmp_path):
    config = tmp_path / "bad.yaml"
    raw = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    raw["frontends"]["orb"]["unknown_parameter"] = 123
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown_parameter"):
        load_settings(config, environ={})


def test_settings_to_dict_is_json_serializable():
    plain = SETTINGS.to_dict()
    assert isinstance(plain["paths"]["project_root"], str)
    assert plain["experiment"]["frontends"] == ["orb", "klt"]
