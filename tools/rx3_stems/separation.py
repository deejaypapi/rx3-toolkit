# SPDX-License-Identifier: MPL-2.0
"""Separation model catalogue and the tunable parameters of each architecture.

`audio-separator` exposes one parameter group per model architecture, and a
model only accepts the group matching its own. The catalogue resolves a model
filename to its architecture so the interface can offer exactly the options
that apply, and so a stored setting for another architecture is never passed
on the command line.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable


CATALOGUE_NAME = "models.json"
SETTINGS_NAME = "separation.json"

DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"
VOCAL_STEM = "Vocals"

QUALITY_MODE = "quality"
NORMAL_MODE = "normal"
QUICK_MODE = "quick"
CUSTOM_MODE = "custom"

LEGACY_MODES = {"fast": QUALITY_MODE}


# ---------------------------------------------------------------------------
# Local ZFTurbo EP17 model
# ---------------------------------------------------------------------------

EP17_MODEL = "model_bs_roformer_ep_17_sdr_9.6568.ckpt"
EP17_CONFIG = "config_bs_roformer_384_8_2_485100.yaml"
EP17_STEMS = ("Vocals", "Drums", "Bass", "Other")
EP17_FRIENDLY_NAME = "ZFTurbo BS-Roformer EP17"
EP17_RELEASE = "v1.0.12"

EP17_MODEL_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/"
    "releases/download/v1.0.12/"
    "model_bs_roformer_ep_17_sdr_9.6568.ckpt"
)

EP17_CONFIG_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/"
    "releases/download/v1.0.12/"
    "config_bs_roformer_384_8_2_485100.yaml"
)


@dataclass(frozen=True)
class Option:
    """One tunable separation parameter, rendered generically by the interface."""

    name: str
    flag: str
    label: str
    help: str
    default: Any
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    upstream: Any = None

    @property
    def implied(self) -> Any:
        """The value that needs no command-line argument."""
        return self.default if self.upstream is None else self.upstream

    def parse(self, value: str) -> Any:
        if self.kind == "integer":
            return int(value)
        if self.kind == "number":
            return float(value)
        return value

    def validate(self, value: Any) -> None:
        if self.kind in ("integer", "number"):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(
                    f"{self.label} must be at least {self.minimum}"
                )
            if self.maximum is not None and value > self.maximum:
                raise ValueError(
                    f"{self.label} must be at most {self.maximum}"
                )


COMMON_OPTIONS: tuple[Option, ...] = (
    Option(
        "normalization",
        "--normalization",
        "Normalization",
        "Max peak amplitude the input and output are normalized to. Keep this "
        "at 1.0: a lower value rescales the vocal out of the source gain "
        "domain and leaves vocal in the instrumental on the deck.",
        1.0,
        "number",
        0.0,
        1.0,
        upstream=0.9,
    ),
    Option(
        "amplification",
        "--amplification",
        "Amplification",
        "Min peak amplitude the input and output are amplified to.",
        0.0,
        "number",
        0.0,
        1.0,
    ),
    Option(
        "invert_spect",
        "--invert_spect",
        "Invert with spectrogram",
        "Derive the stem by spectrogram inversion instead of direct output.",
        False,
        "flag",
    ),
    Option(
        "use_autocast",
        "--use_autocast",
        "Autocast",
        "Faster inference on a GPU. Leave off for the CPU separation used here.",
        False,
        "flag",
    ),
)


ARCHITECTURE_OPTIONS: dict[str, tuple[Option, ...]] = {
    "MDX": (
        Option(
            "mdx_segment_size",
            "--mdx_segment_size",
            "Segment size",
            "Larger uses more memory and may separate better.",
            256,
            "integer",
            32,
            8192,
        ),
        Option(
            "mdx_overlap",
            "--mdx_overlap",
            "Overlap",
            "Overlap between prediction windows. Higher is better and slower.",
            0.25,
            "number",
            0.001,
            0.999,
        ),
        Option(
            "mdx_batch_size",
            "--mdx_batch_size",
            "Batch size",
            "Larger uses more memory and may process slightly faster.",
            1,
            "integer",
            1,
            64,
        ),
        Option(
            "mdx_hop_length",
            "--mdx_hop_length",
            "Hop length",
            "Network stride. Leave at the default unless you know the model.",
            1024,
            "integer",
            64,
            8192,
        ),
        Option(
            "mdx_enable_denoise",
            "--mdx_enable_denoise",
            "Denoise",
            "Denoise while separating. Roughly doubles the processing time.",
            False,
            "flag",
        ),
    ),
    "VR": (
        Option(
            "vr_window_size",
            "--vr_window_size",
            "Window size",
            "1024 is fast and coarse, 320 is slow and finer.",
            512,
            "integer",
            64,
            4096,
        ),
        Option(
            "vr_aggression",
            "--vr_aggression",
            "Aggression",
            "Intensity of the extraction. 5 suits vocals and instrumentals.",
            5,
            "integer",
            -100,
            100,
        ),
        Option(
            "vr_batch_size",
            "--vr_batch_size",
            "Batch size",
            "Larger uses more memory and processes slightly faster.",
            1,
            "integer",
            1,
            64,
        ),
        Option(
            "vr_enable_tta",
            "--vr_enable_tta",
            "Test-time augmentation",
            "Slower, usually cleaner.",
            False,
            "flag",
        ),
        Option(
            "vr_high_end_process",
            "--vr_high_end_process",
            "High-end process",
            "Mirror the missing frequency range into the output.",
            False,
            "flag",
        ),
        Option(
            "vr_enable_post_process",
            "--vr_enable_post_process",
            "Post-process",
            "Identify leftover artifacts in the vocal output.",
            False,
            "flag",
        ),
        Option(
            "vr_post_process_threshold",
            "--vr_post_process_threshold",
            "Post-process threshold",
            "Only used when post-process is on.",
            0.2,
            "number",
            0.1,
            0.3,
        ),
    ),
    "MDXC": (
        Option(
            "mdxc_segment_size",
            "--mdxc_segment_size",
            "Segment size",
            "Larger uses more memory and may separate better.",
            256,
            "integer",
            32,
            8192,
        ),
        Option(
            "mdxc_override_model_segment_size",
            "--mdxc_override_model_segment_size",
            "Override model segment size",
            "Use the segment size above instead of the model's own default.",
            False,
            "flag",
        ),
        Option(
            "mdxc_overlap",
            "--mdxc_overlap",
            "Overlap",
            "Step between prediction windows. Lower stitches the result from "
            "more passes over the audio: better, and considerably slower.",
            8,
            "integer",
            2,
            50,
        ),
        Option(
            "mdxc_batch_size",
            "--mdxc_batch_size",
            "Batch size",
            "Larger uses more memory and may process slightly faster.",
            1,
            "integer",
            1,
            64,
        ),
        Option(
            "mdxc_pitch_shift",
            "--mdxc_pitch_shift",
            "Pitch shift",
            "Shift by semitones while processing. May help very deep or high vocals.",
            0,
            "integer",
            -24,
            24,
        ),
    ),
    "Demucs": (
        Option(
            "demucs_shifts",
            "--demucs_shifts",
            "Shifts",
            "Predictions with random shifts. Higher is better and slower.",
            2,
            "integer",
            1,
            20,
        ),
        Option(
            "demucs_overlap",
            "--demucs_overlap",
            "Overlap",
            "Overlap between prediction windows. Higher is better and slower.",
            0.25,
            "number",
            0.001,
            0.999,
        ),
        Option(
            "demucs_segment_size",
            "--demucs_segment_size",
            "Segment size",
            "Split size for the audio. 'Default' keeps the model's own value.",
            "Default",
            "text",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Roformer models
# ---------------------------------------------------------------------------

ROFORMER_CANDIDATES: tuple[str, ...] = (
    DEFAULT_MODEL,
    "mel_band_roformer_kim_ft_unwa.ckpt",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    EP17_MODEL,
)


MDX_CANDIDATES: tuple[str, ...] = (
    "Kim_Vocal_2.onnx",
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR-MDX-NET_Main_406.onnx",
)


DEMUCS_CANDIDATES: tuple[str, ...] = (
    "htdemucs.yaml",
    "hdemucs_mmi.yaml",
)


QUICK_SHIFTS = 1

NORMAL_OVERLAP = 10

MDX_QUALITY_OVERLAP = 0.5
MDX_QUICK_OVERLAP = 0.1


@dataclass(frozen=True)
class Variant:
    """One model plus its tuning."""

    architecture: str
    candidates: tuple[str, ...]
    summary: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Preset:
    """A named speed/quality trade-off."""

    key: str
    label: str
    torch: Variant
    onnx: Variant

    def resolve(self, *, accelerates_torch: bool = True) -> Variant:
        return self.torch if accelerates_torch else self.onnx


PRESETS: tuple[Preset, ...] = (
    Preset(
        key=QUALITY_MODE,
        label="High quality",
        torch=Variant(
            architecture="MDXC",
            candidates=ROFORMER_CANDIDATES,
            summary="The cleanest, and the slowest.",
        ),
        onnx=Variant(
            architecture="MDX",
            candidates=MDX_CANDIDATES,
            summary=(
                "A lighter model, the only kind this build runs at a usable "
                "speed, over more passes. Nothing is separated above 17.6 kHz."
            ),
            values={"mdx_overlap": MDX_QUALITY_OVERLAP},
        ),
    ),
    Preset(
        key=NORMAL_MODE,
        label="Normal",
        torch=Variant(
            architecture="MDXC",
            candidates=ROFORMER_CANDIDATES,
            summary=(
                "The same model. Noticeably quicker, with almost no difference "
                "in quality."
            ),
            values={"mdxc_overlap": NORMAL_OVERLAP},
        ),
        onnx=Variant(
            architecture="MDX",
            candidates=MDX_CANDIDATES,
            summary=(
                "The same lighter model at its own default overlap. "
                "Nothing is separated above 17.6 kHz."
            ),
        ),
    ),
    Preset(
        key=QUICK_MODE,
        label="Very fast",
        torch=Variant(
            architecture="Demucs",
            candidates=DEMUCS_CANDIDATES,
            summary=(
                "A waveform model. Several times quicker, with noticeably "
                "more instrument left in the vocal."
            ),
            values={"demucs_shifts": QUICK_SHIFTS},
        ),
        onnx=Variant(
            architecture="MDX",
            candidates=MDX_CANDIDATES,
            summary=(
                "The same lighter model, over fewer passes. As quick as "
                "possible, and the least clean."
            ),
        ),
    ),
)


def preset(key: str) -> Preset | None:
    return next(
        (item for item in PRESETS if item.key == key),
        None,
    )


@dataclass(frozen=True)
class Model:
    architecture: str
    name: str
    filename: str
    stems: tuple[str, ...]
    vocal_sdr: float | None
    download_files: tuple[str, ...] = field(default=())

    @property
    def label(self) -> str:
        score = (
            f"SDR {self.vocal_sdr:5.2f}"
            if self.vocal_sdr is not None
            else "SDR    — "
        )

        return (
            f"{score} · {self.architecture:6s} · {self.filename}"
        )

    def local_files(
        self,
        models_directory: pathlib.Path,
    ) -> tuple[pathlib.Path, ...]:
        return tuple(
            models_directory / entry.rsplit("/", 1)[-1]
            for entry in self.download_files
        )

    def is_downloaded(
        self,
        models_directory: pathlib.Path,
    ) -> bool:
        files = self.local_files(
            models_directory
        )

        return bool(files) and all(
            path.is_file()
            for path in files
        )

    def size(
        self,
        models_directory: pathlib.Path,
    ) -> int:
        return sum(
            path.stat().st_size
            for path in self.local_files(models_directory)
            if path.is_file()
        )


@dataclass(frozen=True)
class Catalogue:
    models: tuple[Model, ...] = field(default=())

    def by_filename(
        self,
        filename: str,
    ) -> Model | None:
        for model in self.models:
            if model.filename == filename:
                return model

        return None

    def architecture_of(
        self,
        filename: str,
    ) -> str | None:
        model = self.by_filename(filename)

        return (
            model.architecture
            if model
            else None
        )

    def best_of(
        self,
        architecture: str,
    ) -> Model | None:
        return next(
            (
                model
                for model in self.models
                if model.architecture == architecture
            ),
            None,
        )


def parse_catalogue(
    data: dict[str, Any],
) -> Catalogue:
    """Keep the vocal-capable models of every architecture."""

    models: list[Model] = []

    for architecture, entries in data.items():
        if not isinstance(entries, dict):
            continue

        for name, entry in entries.items():
            if (
                not isinstance(entry, dict)
                or "filename" not in entry
            ):
                continue

            stems = tuple(
                str(stem)
                for stem in entry.get("stems", ())
            )

            if entry.get("filename") == "BS-Roformer-SW.ckpt":
                stems = (
                    "Vocals",
                    "Drums",
                    "Bass",
                    "Other",
                )

            if entry.get("filename") == EP17_MODEL:
                stems = EP17_STEMS

            if VOCAL_STEM.lower() not in {
                stem.lower()
                for stem in stems
            }:
                continue

            score = (
                entry
                .get("scores", {})
                .get("vocals", {})
                .get("SDR")
            )

            downloads = (
                entry.get("download_files")
                or [entry["filename"]]
            )

            models.append(
                Model(
                    architecture=architecture,
                    name=name,
                    filename=str(entry["filename"]),
                    stems=stems,
                    vocal_sdr=(
                        float(score)
                        if isinstance(score, (int, float))
                        else None
                    ),
                    download_files=tuple(
                        str(item)
                        for item in downloads
                    ),
                )
            )

    models.sort(
        key=lambda model: (
            -(model.vocal_sdr or -100.0),
            model.filename,
        )
    )

    return Catalogue(
        models=tuple(models)
    )


def _add_local_ep17(
    catalogue: Catalogue,
    models_directory: pathlib.Path,
) -> Catalogue:
    """Add EP17 to the catalogue even before its first download.

    EP17 is intentionally outside audio-separator's official model catalogue.
    RX3 therefore exposes it explicitly so the user can select it and trigger
    the download when it is first used.
    """

    if catalogue.by_filename(EP17_MODEL) is not None:
        return catalogue

    ep17 = Model(
        architecture="MDXC",
        name=EP17_FRIENDLY_NAME,
        filename=EP17_MODEL,
        stems=EP17_STEMS,
        vocal_sdr=9.6568,
        download_files=(
            EP17_MODEL,
            EP17_CONFIG,
        ),
    )

    models = list(catalogue.models)
    models.append(ep17)

    models.sort(
        key=lambda model: (
            -(model.vocal_sdr or -100.0),
            model.filename,
        )
    )

    return Catalogue(
        models=tuple(models)
    )


def load_catalogue(
    runtime: Any,
    cache: pathlib.Path,
    *,
    refresh: bool = False,
) -> Catalogue:
    """Return the model catalogue and preserve the local EP17 entry."""

    models_directory = pathlib.Path(
        runtime.models
    )

    if not refresh and cache.is_file():
        try:
            catalogue = parse_catalogue(
                json.loads(
                    cache.read_text(
                        encoding="utf-8"
                    )
                )
            )

            return _add_local_ep17(
                catalogue,
                models_directory,
            )

        except (ValueError, OSError):
            pass

    result = subprocess.run(
        [
            str(runtime.separator),
            "--list_models",
            "--list_format=json",
            f"--model_file_dir={runtime.models}",
            "--log_level=error",
        ],
        capture_output=True,
        text=True,
        env=runtime.subprocess_environment(),
    )

    if (
        result.returncode
        or not result.stdout.strip()
    ):
        if cache.is_file():
            catalogue = parse_catalogue(
                json.loads(
                    cache.read_text(
                        encoding="utf-8"
                    )
                )
            )

            return _add_local_ep17(
                catalogue,
                models_directory,
            )

        detail = " ".join(
            (result.stderr or result.stdout).split()
        )[-300:]

        raise RuntimeError(
            f"The model list could not be retrieved: "
            f"{detail or 'no output'}"
        )

    data = json.loads(
        result.stdout
    )

    cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    catalogue = parse_catalogue(
        data
    )

    return _add_local_ep17(
        catalogue,
        models_directory,
    )


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    accelerator: str = "auto"
    values: dict[str, Any] = field(default_factory=dict)
    mode: str = QUALITY_MODE

    def value(
        self,
        option: Option,
    ) -> Any:
        return self.values.get(
            option.name,
            option.default,
        )

    def with_model(
        self,
        model: str,
    ) -> "Settings":
        return replace(
            self,
            model=model,
            mode=CUSTOM_MODE,
        )

    def as_custom(
        self,
    ) -> "Settings":
        return replace(
            self,
            mode=CUSTOM_MODE,
        )

    def with_accelerator(
        self,
        accelerator: str,
    ) -> "Settings":
        return replace(
            self,
            accelerator=accelerator,
        )

    def with_value(
        self,
        option: Option,
        value: Any,
    ) -> "Settings":
        values = dict(
            self.values
        )

        if value == option.default:
            values.pop(
                option.name,
                None,
            )
        else:
            values[option.name] = value

        if values == self.values:
            return self

        return replace(
            self,
            values=values,
            mode=CUSTOM_MODE,
        )

    def options(
        self,
        architecture: str | None,
    ) -> tuple[
        tuple[str, tuple[Option, ...]],
        ...,
    ]:
        groups = [
            ("Common", COMMON_OPTIONS)
        ]

        if (
            architecture
            and architecture in ARCHITECTURE_OPTIONS
        ):
            groups.append(
                (
                    f"{architecture} architecture",
                    ARCHITECTURE_OPTIONS[architecture],
                )
            )

        return tuple(groups)

    def arguments(
        self,
        architecture: str | None,
    ) -> list[str]:
        arguments = [
            f"--model_filename={self.model}"
        ]

        for _, options in self.options(
            architecture
        ):
            for option in options:
                value = self.value(option)

                option.validate(
                    value
                )

                if value == option.implied:
                    continue

                if option.kind == "flag":
                    if value:
                        arguments.append(
                            option.flag
                        )
                else:
                    arguments.append(
                        f"{option.flag}={value}"
                    )

        return arguments


def resolve_preset_model(
    variant: Variant,
    catalogue: Catalogue,
) -> str:
    """The model a preset variant means on this installation."""

    for candidate in variant.candidates:
        if catalogue.by_filename(
            candidate
        ) is not None:
            return candidate

    best = catalogue.best_of(
        variant.architecture
    )

    if best is not None:
        return best.filename

    return (
        variant.candidates[0]
        if variant.candidates
        else DEFAULT_MODEL
    )


def apply_preset(
    settings: Settings,
    item: Preset,
    catalogue: Catalogue,
    *,
    accelerates_torch: bool = True,
) -> Settings:
    """Return settings reconfigured for one preset."""

    variant = item.resolve(
        accelerates_torch=accelerates_torch
    )

    model = resolve_preset_model(
        variant,
        catalogue,
    )

    architecture = (
        catalogue.architecture_of(model)
        or variant.architecture
    )

    applicable = {
        option.name: option
        for _, options in Settings().options(
            architecture
        )
        for option in options
    }

    values = {
        name: value
        for name, value in variant.values.items()
        if (
            name in applicable
            and value != applicable[name].default
        )
    }

    return replace(
        settings,
        model=model,
        values=values,
        mode=item.key,
    )


def load_settings(
    path: pathlib.Path,
) -> Settings:
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (ValueError, OSError):
        return Settings()

    model = data.get(
        "model"
    )

    accelerator = data.get(
        "accelerator"
    )

    values = data.get(
        "values"
    )

    mode = LEGACY_MODES.get(
        data.get("mode"),
        data.get("mode"),
    )

    known = set(
        known_option_names()
    )

    return Settings(
        model=(
            model
            if isinstance(model, str)
            and model
            else DEFAULT_MODEL
        ),
        accelerator=(
            accelerator
            if isinstance(accelerator, str)
            and accelerator
            else "auto"
        ),
        values={
            key: value
            for key, value in (
                values or {}
            ).items()
            if key in known
        }
        if isinstance(values, dict)
        else {},
        mode=(
            mode
            if mode in {
                item.key
                for item in PRESETS
            } | {CUSTOM_MODE}
            else CUSTOM_MODE
        ),
    )


def save_settings(
    path: pathlib.Path,
    settings: Settings,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "model": settings.model,
                "accelerator": settings.accelerator,
                "mode": settings.mode,
                "values": settings.values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _ensure_ep17_download_registry(
    models_directory: pathlib.Path,
) -> None:
    """Register EP17 in audio-separator's local download registry.

    audio-separator builds its supported model list from download_checks.json.
    EP17 is not present in the upstream registry, so add only this one model
    under the existing RoFormer download list.
    """

    registry_path = (
        models_directory
        / "download_checks.json"
    )

    data: dict[str, Any] = {}

    if registry_path.is_file():
        try:
            loaded = json.loads(
                registry_path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(loaded, dict):
                data = loaded

        except (ValueError, OSError):
            data = {}

    roformer = data.get(
        "roformer_download_list"
    )

    if not isinstance(roformer, dict):
        roformer = {}

    expected = {
        EP17_MODEL: EP17_CONFIG,
    }

    if roformer.get(
        EP17_FRIENDLY_NAME
    ) == expected:
        return

    roformer[EP17_FRIENDLY_NAME] = expected
    data["roformer_download_list"] = roformer

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path.write_text(
        json.dumps(
            data,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _download_ep17_file(
    url: str,
    destination: pathlib.Path,
    progress: Callable[[str], None],
) -> None:
    """Download one EP17 asset atomically into the model directory."""

    if destination.is_file():
        return

    temporary = destination.with_suffix(
        destination.suffix + ".download"
    )

    if temporary.exists():
        temporary.unlink()

    progress(
        f"Downloading {destination.name}…"
    )

    try:
        urllib.request.urlretrieve(
            url,
            temporary,
        )
        temporary.replace(
            destination
        )

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def download_model(
    runtime: Any,
    model: Model,
    progress: Callable[
        [str],
        None,
    ] = lambda message: None,
) -> None:
    """Fetch one model's files, leaving every other model untouched."""

    models_directory = pathlib.Path(
        runtime.models
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # EP17 is outside audio-separator's official catalogue.
    # Register it first so the installed audio-separator understands the model.
    if model.filename == EP17_MODEL:
        _ensure_ep17_download_registry(
            models_directory
        )

        checkpoint = (
            models_directory
            / EP17_MODEL
        )

        config = (
            models_directory
            / EP17_CONFIG
        )

        _download_ep17_file(
            EP17_MODEL_URL,
            checkpoint,
            progress,
        )

        _download_ep17_file(
            EP17_CONFIG_URL,
            config,
            progress,
        )

        if not checkpoint.is_file():
            raise RuntimeError(
                f"{EP17_MODEL} could not be downloaded."
            )

        if not config.is_file():
            raise RuntimeError(
                f"{EP17_CONFIG} could not be downloaded."
            )

        progress(
            f"{EP17_MODEL} downloaded."
        )

        return

    progress(
        f"Downloading {model.filename}…"
    )

    process = subprocess.Popen(
        [
            str(runtime.separator),
            "--download_model_only",
            f"--model_filename={model.filename}",
            f"--model_file_dir={models_directory}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=runtime.subprocess_environment(),
    )

    assert process.stdout is not None

    tail: list[str] = []

    for line in process.stdout:
        line = line.strip()

        if line:
            tail = (
                tail + [line]
            )[-8:]

            progress(
                f"{model.filename}: {line[:140]}"
            )

    if process.wait():
        raise RuntimeError(
            f"{model.filename} could not be downloaded: "
            f"{' / '.join(tail)[-400:] or 'no output'}"
        )

    if not model.is_downloaded(
        models_directory
    ):
        missing = [
            path.name
            for path in model.local_files(
                models_directory
            )
            if not path.is_file()
        ]

        raise RuntimeError(
            f"{model.filename} is still incomplete: "
            f"missing {', '.join(missing)}"
        )

    progress(
        f"{model.filename} downloaded."
    )


def delete_model(
    models_directory: pathlib.Path,
    model: Model,
) -> int:
    """Remove one model's files and return how many bytes were freed."""

    freed = 0

    for path in model.local_files(
        models_directory
    ):
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()

    return freed


def normalization_option() -> Option:
    return next(
        option
        for option in COMMON_OPTIONS
        if option.name == "normalization"
    )


def input_normalization(
    settings: Settings,
    architecture: str | None,
) -> float | None:
    """The peak the separator scales the mix to before inference, if it does."""

    if architecture not in (
        "MDX",
        "MDXC",
    ):
        return None

    return float(
        settings.value(
            normalization_option()
        )
    )


def known_option_names() -> Iterable[str]:
    for options in (
        COMMON_OPTIONS,
        *ARCHITECTURE_OPTIONS.values(),
    ):
        for option in options:
            yield option.name