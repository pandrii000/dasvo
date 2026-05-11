from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Mapping

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs" / "default.yaml"


def project_path(value: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


ProjectPath = Annotated[Path, AfterValidator(project_path)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathSettings(ConfigModel):
    project_root: ProjectPath = Field(default=PROJECT_ROOT)
    workspace_root: ProjectPath
    assets_root: ProjectPath
    data_root: ProjectPath
    outputs_root: ProjectPath
    trajectories_root: ProjectPath
    robustness_outputs_root: ProjectPath
    tables_root: ProjectPath
    figures_root: ProjectPath
    logs_root: ProjectPath
    config_snapshots_root: ProjectPath
    paper_project_root: ProjectPath
    paper_assets_root: ProjectPath
    paper_figures_root: ProjectPath
    paper_tables_root: ProjectPath


class DatasetSettings(ConfigModel):
    name: str
    sequences_file: ProjectPath
    sequence_suites: dict[str, ProjectPath]
    manifest_file: ProjectPath
    image_dir: str
    depth_dir: str
    pose_file: str
    image_glob: str
    depth_glob: str
    require_complete: bool


class CameraSettings(ConfigModel):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_min_m: float
    depth_max_m: float

    @property
    def intrinsics(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0))


class DownloadSettings(ConfigModel):
    source: str
    extract: bool
    clean: bool
    overwrite: bool
    progress: bool
    chunk_size_bytes: int
    hf_base_url: str
    s3_endpoint_url: str
    s3_bucket: str
    archives: dict[str, str]
    data_dirs: dict[str, str]
    pose_file: str

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        source = value.lower()
        if source in {"hf", "huggingface"}:
            return "hf"
        if source in {"s3", "cmu"}:
            return "s3"
        raise ValueError("download.source must be 'hf' or 's3'")


class OrbSettings(ConfigModel):
    nfeatures: int
    scale_factor: float
    nlevels: int
    edge_threshold: int
    first_level: int
    wta_k: int
    score_type: str
    patch_size: int
    fast_threshold: int
    matcher: str
    ratio_test: float
    cross_check: bool
    min_matches: int


class KltSettings(ConfigModel):
    max_corners: int
    quality_level: float
    min_distance: float
    block_size: int
    use_harris_detector: bool
    harris_k: float
    win_size: tuple[int, int]
    max_level: int
    criteria_count: int
    criteria_eps: float
    redetect_below: int


class FrontendSettings(ConfigModel):
    orb: OrbSettings
    klt: KltSettings


class EssentialSettings(ConfigModel):
    min_correspondences: int
    ransac_probability: float
    ransac_threshold_px: float
    recover_pose_distance_threshold_m: float | None = None


class PnpSettings(ConfigModel):
    min_correspondences: int
    min_inliers: int
    iterations_count: int
    reprojection_error_px: float
    confidence: float
    solver: str
    max_translation_norm_m: float


class BackendSettings(ConfigModel):
    essential: EssentialSettings
    pnp: PnpSettings


class DegradationPreset(ConfigModel):
    kind: str
    apply_to: str
    enabled: bool
    parameters: dict[str, Any] = Field(default_factory=dict)


class DegradationSettings(ConfigModel):
    default: str
    presets: dict[str, DegradationPreset]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.presets)

    @model_validator(mode="after")
    def validate_default(self) -> DegradationSettings:
        if self.default not in self.presets:
            raise ValueError("degradations.default must exist in degradations.presets")
        return self


class PrecomputedSettings(ConfigModel):
    root: ProjectPath
    frontend_root: ProjectPath
    backend_root: ProjectPath
    frontend_roots: dict[str, ProjectPath]
    backend_roots: dict[str, ProjectPath]
    correspondences_root: ProjectPath
    depth_samples_root: ProjectPath


class ExperimentSettings(ConfigModel):
    frontends: tuple[str, ...]
    backends: tuple[str, ...]
    frame_strides: tuple[int, ...]
    degradations: tuple[str, ...]
    max_virtual_frames: int | None = None
    random_seed: int
    num_workers: int
    save_run_config: bool


class EvaluationSettings(ConfigModel):
    metrics: tuple[str, ...]
    rpe_delta: int
    monocular_scale_align_backends: tuple[str, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    pose_sample_sizes: tuple[int, ...]
    segment_sample_sizes: tuple[int, ...]


class PublicationSettings(ConfigModel):
    copy_to_paper_assets: bool
    figure_names: tuple[str, ...]
    table_names: tuple[str, ...]
    hardware_devices: tuple[str, ...]


class DasvoSettings(ConfigModel):
    config_file: ProjectPath
    paths: PathSettings
    dataset: DatasetSettings
    camera: CameraSettings
    download: DownloadSettings
    frontends: FrontendSettings
    backends: BackendSettings
    degradations: DegradationSettings
    precomputed: PrecomputedSettings
    experiment: ExperimentSettings
    evaluation: EvaluationSettings
    publication: PublicationSettings

    @model_validator(mode="after")
    def validate_cross_references(self) -> DasvoSettings:
        missing_degradations = set(self.experiment.degradations) - set(self.degradations.presets)
        if missing_degradations:
            raise ValueError(f"unknown experiment degradations: {sorted(missing_degradations)}")

        missing_frontends = set(self.experiment.frontends) - set(self.precomputed.frontend_roots)
        if missing_frontends:
            raise ValueError(f"missing precomputed frontend roots: {sorted(missing_frontends)}")

        missing_backends = set(self.experiment.backends) - set(self.precomputed.backend_roots)
        if missing_backends:
            raise ValueError(f"missing precomputed backend roots: {sorted(missing_backends)}")

        if len(self.evaluation.pose_sample_sizes) != len(self.evaluation.segment_sample_sizes):
            raise ValueError("evaluation pose_sample_sizes and segment_sample_sizes must match")

        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "DASVO_ASSETS_ROOT": ("paths", "assets_root"),
    "DASVO_DATA_ROOT": ("paths", "data_root"),
    "DASVO_OUTPUTS_ROOT": ("paths", "outputs_root"),
    "DASVO_TRAJECTORIES_ROOT": ("paths", "trajectories_root"),
    "DASVO_ROBUSTNESS_OUTPUTS_ROOT": ("paths", "robustness_outputs_root"),
    "DASVO_TABLES_ROOT": ("paths", "tables_root"),
    "DASVO_FIGURES_ROOT": ("paths", "figures_root"),
    "DASVO_PAPER_PROJECT_ROOT": ("paths", "paper_project_root"),
    "DASVO_PAPER_ASSETS_ROOT": ("paths", "paper_assets_root"),
    "DASVO_PAPER_FIGURES_ROOT": ("paths", "paper_figures_root"),
    "DASVO_PAPER_TABLES_ROOT": ("paths", "paper_tables_root"),
    "DASVO_SEQUENCES_FILE": ("dataset", "sequences_file"),
    "DASVO_MANIFEST_FILE": ("dataset", "manifest_file"),
    "DASVO_DOWNLOAD_SOURCE": ("download", "source"),
    "DASVO_DOWNLOAD_CHUNK_SIZE": ("download", "chunk_size_bytes"),
    "DASVO_PRECOMPUTED_ROOT": ("precomputed", "root"),
    "DASVO_FRONTEND_ASSETS_ROOT": ("precomputed", "frontend_root"),
    "DASVO_BACKEND_ASSETS_ROOT": ("precomputed", "backend_root"),
    "DASVO_NUM_WORKERS": ("experiment", "num_workers"),
    "DASVO_RANDOM_SEED": ("experiment", "random_seed"),
}


def load_settings(
    config_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> DasvoSettings:
    env = os.environ if environ is None else environ
    path = project_path(Path(config_file or env.get("DASVO_CONFIG_FILE", DEFAULT_CONFIG_FILE)))
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    raw["config_file"] = path
    for env_name, keys in ENV_OVERRIDES.items():
        if env_name in env:
            section = raw
            for key in keys[:-1]:
                section = section.setdefault(key, {})
            section[keys[-1]] = env[env_name]

    return DasvoSettings.model_validate(raw)


SETTINGS = load_settings()
CONFIG_FILE = SETTINGS.config_file

WORKSPACE_ROOT = SETTINGS.paths.workspace_root
ASSETS_ROOT = SETTINGS.paths.assets_root
DATA_ROOT = SETTINGS.paths.data_root
SEQUENCES_FILE = SETTINGS.dataset.sequences_file
SEQUENCE_SUITES = SETTINGS.dataset.sequence_suites
MANIFEST_FILE = SETTINGS.dataset.manifest_file

OUTPUTS_ROOT = SETTINGS.paths.outputs_root
TRAJECTORIES_ROOT = SETTINGS.paths.trajectories_root
ROBUSTNESS_OUTPUTS_ROOT = SETTINGS.paths.robustness_outputs_root
TABLES_ROOT = SETTINGS.paths.tables_root
FIGURES_ROOT = SETTINGS.paths.figures_root
LOGS_ROOT = SETTINGS.paths.logs_root
CONFIG_SNAPSHOTS_ROOT = SETTINGS.paths.config_snapshots_root

PAPER_PROJECT_ROOT = SETTINGS.paths.paper_project_root
PAPER_ASSETS_ROOT = SETTINGS.paths.paper_assets_root
PAPER_FIGURES_ROOT = SETTINGS.paths.paper_figures_root
PAPER_TABLES_ROOT = SETTINGS.paths.paper_tables_root

PRECOMPUTED_ROOT = SETTINGS.precomputed.root
FRONTEND_ASSETS_ROOT = SETTINGS.precomputed.frontend_root
BACKEND_ASSETS_ROOT = SETTINGS.precomputed.backend_root
FRONTEND_ASSET_ROOTS = SETTINGS.precomputed.frontend_roots
BACKEND_ASSET_ROOTS = SETTINGS.precomputed.backend_roots
CORRESPONDENCES_ROOT = SETTINGS.precomputed.correspondences_root
DEPTH_SAMPLES_ROOT = SETTINGS.precomputed.depth_samples_root

DOWNLOAD_SOURCE = SETTINGS.download.source
FRONTENDS = SETTINGS.experiment.frontends
BACKENDS = SETTINGS.experiment.backends
FRAME_STRIDES = SETTINGS.experiment.frame_strides
DEGRADATION_PRESETS = SETTINGS.degradations.presets
DEGRADATIONS = SETTINGS.experiment.degradations
