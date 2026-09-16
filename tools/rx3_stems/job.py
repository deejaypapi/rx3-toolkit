# SPDX-License-Identifier: MPL-2.0
"""Generation pipeline: separate a playlist locally and write `RX3_STEMS`."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from tools.rx3_stems.estimate import Estimator
from tools.rx3_stems.provisioning import Acceleration, Runtime, resolve_acceleration
from tools.rx3_stems.rekordbox import Collection, Playlist, Track, export_stem
from tools.rx3_stems.separation import VOCAL_STEM, Settings, input_normalization
from tools.rx3_stems.sidecar import write_sidecar


MANIFEST_NAME = "rx3-stems-manifest.json"
OUTPUT_NAME = "RX3_STEMS"
SIDECAR_SUFFIX = ".rx3stem"
BASS_SUFFIX = ".rx3bass"
DRUMS_SUFFIX = ".rx3drums"

MINIMUM_SIDECAR_BYTES = 64
PERCENT = re.compile(r"(\d{1,3})%")

PERCENT_WINDOW = 96
TRANSCRIPT_LIMIT = 16384
REPORTED_DETAIL = 600

HEAD_LIMIT = 8192
CPU_FALLBACK = "No hardware acceleration could be configured"


def failure_detail(transcript: str) -> str:
    """Summarise a separator transcript, dropping its progress-bar noise."""
    lines = [
        line.strip()
        for line in transcript.replace("\r", "\n").splitlines()
        if line.strip() and "%|" not in line
    ]
    if not lines:
        return "no output"

    detail = " / ".join(lines[-6:])
    return detail[-REPORTED_DETAIL:]


@dataclass(frozen=True)
class TrackResult:
    track_id: str
    artist: str
    title: str
    source_file: str
    sidecar: str
    size: int
    status: str

    sidecars: tuple[str, ...] = ()

    gain: float = 1.0
    clipped: int = 0

    delay: int = 0

    def as_manifest_entry(self) -> dict[str, object]:
        entry: dict[str, object] = {
            "trackId": self.track_id,
            "artist": self.artist,
            "title": self.title,
            "sourceFile": self.source_file,
            "sidecar": self.sidecar,
            "bytes": self.size,
            "status": self.status,
            "gainCorrection": round(self.gain, 6),
            "clippedSamples": self.clipped,
            "encoderDelayFrames": self.delay,
        }

        if self.sidecars:
            entry["sidecars"] = list(self.sidecars)

        return entry


@dataclass(frozen=True)
class TrackError:
    track: str
    error: str


@dataclass(frozen=True)
class JobState:
    state: str = "idle"
    stage: str = ""
    current: str = ""
    progress: int = 0
    track_progress: int = 0

    completed: int = 0

    position: int = 0
    total: int = 0

    eta_seconds: float | None = None
    elapsed_seconds: float = 0.0

    output: pathlib.Path | None = None
    manifest: pathlib.Path | None = None
    fatal: str = ""

    results: tuple[TrackResult, ...] = field(default=())
    errors: tuple[TrackError, ...] = field(default=())

    notices: tuple[str, ...] = field(default=())


class Cancelled(Exception):
    """The operator stopped the job."""


class StemJob:
    """Run one playlist through separation and sidecar encoding."""

    def __init__(
        self,
        runtime: Runtime,
        collection: Collection,
        playlist: Playlist,
        output_root: pathlib.Path,
        *,
        settings: Settings | None = None,
        architecture: str | None = None,
        acceleration: Acceleration | None = None,
        estimator: Estimator | None = None,
        observer: Callable[[JobState], None] = lambda state: None,
    ) -> None:
        self.runtime = runtime
        self.collection = collection
        self.playlist = playlist
        self.output_root = output_root
        self.settings = settings or Settings()
        self.architecture = architecture
        self.acceleration = acceleration or resolve_acceleration(
            self.settings.accelerator
        )
        self.estimator = estimator or Estimator(
            architecture, self.acceleration.key
        )
        self.observer = observer
        self._lock = threading.Lock()
        self._state = JobState()
        self._process: subprocess.Popen[str] | None = None
        self._cancelled = False
        self._started = 0.0

        self._current_audio = 0.0
        self._later_audio = 0.0

        # Demucs models produce their native multi-stem output.
        #
        # BS-Roformer-SW is also a multi-stem model.
        #
        # ZFTurbo BS-Roformer MUSDB18HQ is a local 4-stem model:
        # Vocals / Drums / Bass / Other.
        #
        # RX3 only needs Vocals / Drums / Bass. The Other stem is ignored
        # by _find_multi_stems() and is automatically removed with the
        # temporary workspace after each track.
        self._multi_stem = (
            isinstance(self.architecture, str)
            and (
                self.architecture.strip().lower() == "demucs"
                or self.settings.model == "BS-Roformer-SW.ckpt"
                or self.settings.model
                == "model_bs_roformer_ep_17_sdr_9.6568.ckpt"
            )
        )

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    def _update(self, **values: object) -> None:
        with self._lock:
            self._state = replace(self._state, **values)
            snapshot = self._state
        self.observer(snapshot)

    def cancel(self) -> None:
        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)

    def _checkpoint(self) -> None:
        if self._cancelled:
            raise Cancelled("Cancelled")

    def _timings(self, track_fraction: float) -> dict[str, object]:
        """Elapsed time, and the computation the rest of the playlist needs."""
        remaining = (
            self._later_audio
            + self._current_audio * max(0.0, 1.0 - track_fraction)
        )
        elapsed = time.monotonic() - self._started if self._started else 0.0

        return {
            "eta_seconds": self.estimator.remaining(remaining),
            "elapsed_seconds": elapsed,
        }

    def _notice(self, message: str) -> None:
        """Record a condition once, however many tracks reproduce it."""
        if message not in self._state.notices:
            self._update(notices=self._state.notices + (message,))

    def _note_inference_device(self, head: str) -> None:
        """Say so when an accelerated runtime silently ran on the CPU."""
        if self.acceleration.key == "cpu" or CPU_FALLBACK not in head:
            return

        self._notice(
            f"{self.acceleration.label} was selected but the separator found no "
            "usable device and ran on the CPU. Install the separation runtime "
            "again for this accelerator, which rebuilds PyTorch for it."
        )

    @staticmethod
    def _stem_kind(path: pathlib.Path) -> str | None:
        """Identify a separated WAV by the stem name used by audio-separator."""
        name = path.stem.casefold()

        if re.search(
            r"(?:^|[\s_\-.(\[])vocals(?:$|[\s_\-.)\]])",
            name,
        ):
            return "vocals"

        if re.search(
            r"(?:^|[\s_\-.(\[])drums(?:$|[\s_\-.)\]])",
            name,
        ):
            return "drums"

        if re.search(
            r"(?:^|[\s_\-.(\[])bass(?:$|[\s_\-.)\]])",
            name,
        ):
            return "bass"

        return None

    def _find_multi_stems(
        self,
        workspace: pathlib.Path,
        transcript: str,
    ) -> dict[str, pathlib.Path]:
        """Find Vocals, Bass and Drums in a multi-stem separator workspace."""
        candidates = sorted(workspace.rglob("*.wav"))

        found: dict[str, pathlib.Path] = {}

        for candidate in candidates:
            kind = self._stem_kind(candidate)
            if kind is None:
                continue

            found.setdefault(kind, candidate)

        required = ("vocals", "bass", "drums")
        missing = [kind for kind in required if kind not in found]

        if missing:
            available = ", ".join(
                sorted(
                    {
                        kind
                        for path in candidates
                        if (kind := self._stem_kind(path)) is not None
                    }
                )
            ) or "none"

            raise RuntimeError(
                "Expected Vocals, Bass and Drums from the selected separator, "
                f"but missing: {', '.join(missing)}. "
                f"Detected stems: {available}. "
                f"{failure_detail(transcript)}"
            )

        return found

    def _separate(
        self,
        source: pathlib.Path,
        workspace: pathlib.Path,
        index: int,
        total: int,
    ) -> pathlib.Path | dict[str, pathlib.Path]:
        if self.runtime.separator is None:
            raise RuntimeError("audio-separator is not installed")

        command = [
            str(self.runtime.separator),
            str(source),
            f"--model_file_dir={self.runtime.models}",
            f"--output_dir={workspace}",
            "--output_format=WAV",
        ]

        # All multi-stem models are allowed to produce their native outputs.
        # Single-stem models keep the original vocal-only behavior.
        if not self._multi_stem:
            command.append(f"--single_stem={VOCAL_STEM}")

        command.extend(
            [
                *self.settings.arguments(self.architecture),
                *self.acceleration.separation_flags,
            ]
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self.runtime.subprocess_environment(),
            )
        except OSError as error:
            raise RuntimeError(
                f"audio-separator could not be started from "
                f"{self.runtime.separator}: {error}. "
                "Install the separation runtime again."
            ) from error

        self._process = process
        assert process.stdout is not None

        window = ""
        transcript = ""
        head = ""
        last = -1

        while True:
            character = process.stdout.read(1)

            if not character:
                break

            window = (window + character)[-PERCENT_WINDOW:]
            transcript = (transcript + character)[-TRANSCRIPT_LIMIT:]

            if len(head) < HEAD_LIMIT:
                head += character

            if character == "%":
                match = PERCENT.search(window)
                if match:
                    track_progress = min(int(match.group(1)), 95)

                    if track_progress != last:
                        overall = round(
                            (
                                (index + track_progress / 100)
                                / total
                            )
                            * 100
                        )

                        self._update(
                            track_progress=track_progress,
                            progress=overall,
                            **self._timings(track_progress / 100),
                        )
                        last = track_progress

            if self._cancelled:
                break

        return_code = process.wait()
        self._process = None

        self._note_inference_device(head)
        self._checkpoint()

        if return_code:
            raise RuntimeError(
                f"audio-separator exited with code {return_code}: "
                f"{failure_detail(transcript)}"
            )

        candidates = sorted(workspace.rglob("*.wav"))

        if self._multi_stem:
            return self._find_multi_stems(workspace, transcript)

        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one separated stem, found {len(candidates)}: "
                f"{failure_detail(transcript)}"
            )

        return candidates[0]

    def _encode_one(
        self,
        separated: pathlib.Path,
        destination: pathlib.Path,
        source: pathlib.Path,
    ):
        """Encode one separated WAV into an RX3 sidecar in the temp workspace."""
        local = destination.parent / destination.name

        return write_sidecar(
            separated,
            local,
            ffmpeg=self.runtime.ffmpeg or "ffmpeg",
            sample_format="s16",
            match_full=source,
            separator_normalization=input_normalization(
                self.settings,
                self.architecture,
            ),
        )

    def _process_track(
        self,
        track: Track,
        index: int,
        total: int,
        output: pathlib.Path,
        used_names: dict[str, pathlib.Path],
    ) -> TrackResult:
        source = track.location

        if not source.is_file():
            raise FileNotFoundError("Source file not found")

        base = export_stem(source.stem)

        collision = used_names.get(base.casefold())

        if collision is not None and collision != source:
            raise ValueError(
                f"Ambiguous filename with {collision.name}: both are exported as "
                f"{base}, and the RX3 load interface cannot distinguish them"
            )

        used_names[base.casefold()] = source

        vocal_destination = output / f"{base}{SIDECAR_SUFFIX}"
        bass_destination = output / f"{base}{BASS_SUFFIX}"
        drums_destination = output / f"{base}{DRUMS_SUFFIX}"

        if self._multi_stem:
            destinations = {
                "vocals": vocal_destination,
                "bass": bass_destination,
                "drums": drums_destination,
            }

            complete_existing = all(
                path.is_file()
                and path.stat().st_size > MINIMUM_SIDECAR_BYTES
                for path in destinations.values()
            )

            if complete_existing:
                self._update(
                    stage="Already generated",
                    track_progress=100,
                )

                return TrackResult(
                    track_id=track.track_id,
                    artist=track.artist,
                    title=track.title,
                    source_file=source.name,
                    sidecar=vocal_destination.name,
                    size=vocal_destination.stat().st_size,
                    status="existing",
                    sidecars=(
                        vocal_destination.name,
                        bass_destination.name,
                        drums_destination.name,
                    ),
                )

            status = "created"
            gain, clipped, delay = 1.0, 0, 0

            partials = {
                kind: path.with_suffix(path.suffix + ".partial")
                for kind, path in destinations.items()
            }

            try:
                with tempfile.TemporaryDirectory(
                    prefix="rx3-stem-"
                ) as directory:
                    workspace = pathlib.Path(directory)

                    self._update(
                        stage="Stem separation",
                        track_progress=1,
                    )

                    separated = self._separate(
                        source,
                        workspace,
                        index,
                        total,
                    )

                    assert isinstance(separated, dict)

                    self._update(
                        stage="Sidecar encoding",
                        track_progress=96,
                    )

                    encoded_info: dict[str, object] = {}

                    ordered = (
                        ("vocals", vocal_destination),
                        ("bass", bass_destination),
                        ("drums", drums_destination),
                    )

                    for kind, destination in ordered:
                        encoded = self._encode_one(
                            separated[kind],
                            workspace / destination.name,
                            source,
                        )

                        encoded_info[kind] = encoded

                        if not encoded.aligned:
                            self._notice(
                                f"{destination.name}: the encoder padding of "
                                f"{source.name} could not be measured, so the stem "
                                "stays on the separator's timeline. If the deck "
                                "leaves vocal in the instrumental, convert the "
                                "source to WAV or FLAC and generate it again."
                            )

                        if encoded.clipped:
                            self._notice(
                                f"{destination.name}: {encoded.clipped} sample(s) "
                                "exceeded full scale and were clipped by the "
                                "s16 sidecar."
                            )

                    if not output.is_dir():
                        raise RuntimeError(
                            "The destination was unmounted during processing"
                        )

                    for kind, destination in ordered:
                        local = workspace / destination.name
                        shutil.copyfile(local, partials[kind])

                    for kind, destination in ordered:
                        partials[kind].replace(destination)

                    vocal_encoded = encoded_info["vocals"]

                    gain = vocal_encoded.gain
                    clipped = vocal_encoded.clipped
                    delay = vocal_encoded.delay

            except BaseException:
                for partial in partials.values():
                    partial.unlink(missing_ok=True)
                raise

            return TrackResult(
                track_id=track.track_id,
                artist=track.artist,
                title=track.title,
                source_file=source.name,
                sidecar=vocal_destination.name,
                size=vocal_destination.stat().st_size,
                status=status,
                sidecars=(
                    vocal_destination.name,
                    bass_destination.name,
                    drums_destination.name,
                ),
                gain=gain,
                clipped=clipped,
                delay=delay,
            )

        # ------------------------------------------------------------------
        # Original single-vocal path.
        # ------------------------------------------------------------------

        destination = vocal_destination
        gain, clipped, delay = 1.0, 0, 0

        if (
            destination.is_file()
            and destination.stat().st_size > MINIMUM_SIDECAR_BYTES
        ):
            self._update(
                stage="Already generated",
                track_progress=100,
            )
            status = "existing"

        else:
            status = "created"
            partial = destination.with_suffix(
                f"{SIDECAR_SUFFIX}.partial"
            )

            try:
                with tempfile.TemporaryDirectory(
                    prefix="rx3-stem-"
                ) as directory:
                    workspace = pathlib.Path(directory)

                    self._update(
                        stage="Vocal separation",
                        track_progress=1,
                    )

                    vocals = self._separate(
                        source,
                        workspace,
                        index,
                        total,
                    )

                    assert isinstance(vocals, pathlib.Path)

                    self._update(
                        stage="Sidecar encoding",
                        track_progress=96,
                    )

                    local = workspace / destination.name

                    encoded = write_sidecar(
                        vocals,
                        local,
                        ffmpeg=self.runtime.ffmpeg or "ffmpeg",
                        sample_format="s16",
                        match_full=source,
                        separator_normalization=input_normalization(
                            self.settings,
                            self.architecture,
                        ),
                    )

                    gain = encoded.gain
                    clipped = encoded.clipped
                    delay = encoded.delay

                    if not encoded.aligned:
                        self._notice(
                            f"{destination.name}: the encoder padding of "
                            f"{source.name} could not be measured, so the stem "
                            "stays on the separator's timeline. If the deck "
                            "leaves vocal in the instrumental, convert the "
                            "source to WAV or FLAC and generate it again."
                        )

                    if clipped:
                        self._notice(
                            f"{destination.name}: {clipped} sample(s) exceeded "
                            "full scale and were clipped by the s16 sidecar."
                        )

                    if not output.is_dir():
                        raise RuntimeError(
                            "The destination was unmounted during processing"
                        )

                    shutil.copyfile(local, partial)
                    partial.replace(destination)

            except BaseException:
                partial.unlink(missing_ok=True)
                raise

        return TrackResult(
            track_id=track.track_id,
            artist=track.artist,
            title=track.title,
            source_file=source.name,
            sidecar=destination.name,
            size=destination.stat().st_size,
            status=status,
            sidecars=(destination.name,),
            gain=gain,
            clipped=clipped,
            delay=delay,
        )

    def run(self) -> JobState:
        """Process every track, recording per-track failures without stopping."""
        try:
            tracks = self.playlist.tracks

            if not tracks:
                raise ValueError("The playlist is empty")

            output = self.output_root / OUTPUT_NAME

            try:
                output.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            except OSError as error:
                raise RuntimeError(
                    f"{output} could not be created: "
                    f"{error.strerror or error}. "
                    "Choose a writable output folder or mounted USB drive."
                ) from error

            self._started = time.monotonic()

            durations = [
                float(max(0, track.duration or 0))
                for track in tracks
            ]

            self._current_audio = 0.0
            self._later_audio = sum(durations)

            self._update(
                state="running",
                total=len(tracks),
                output=output,
                stage="Preparing",
                **self._timings(0.0),
            )

            results: list[TrackResult] = []
            errors: list[TrackError] = []
            used_names: dict[str, pathlib.Path] = {}

            for index, track in enumerate(tracks):
                self._checkpoint()

                if not output.is_dir():
                    raise RuntimeError(
                        f"The output directory disappeared: {output}. "
                        "The USB drive was most likely unmounted."
                    )

                self._current_audio = durations[index]
                self._later_audio = sum(
                    durations[index + 1:]
                )

                self._update(
                    current=track.label,
                    completed=index,
                    position=index + 1,
                    track_progress=0,
                    progress=round(
                        index / len(tracks) * 100
                    ),
                    stage="Checking",
                    **self._timings(0.0),
                )

                track_started = time.monotonic()

                try:
                    result = self._process_track(
                        track,
                        index,
                        len(tracks),
                        output,
                        used_names,
                    )

                    results.append(result)

                    if result.status == "created":
                        self.estimator.observe(
                            durations[index],
                            time.monotonic() - track_started,
                        )

                    self._update(results=tuple(results))

                except Cancelled:
                    raise

                except Exception as error:
                    errors.append(
                        TrackError(
                            track=track.label,
                            error=str(error),
                        )
                    )
                    self._update(errors=tuple(errors))

                self._current_audio = 0.0

                self._update(
                    completed=index + 1,
                    track_progress=100,
                    progress=round(
                        (index + 1) / len(tracks) * 100
                    ),
                    **self._timings(1.0),
                )

            manifest = self.output_root / MANIFEST_NAME

            manifest.write_text(
                json.dumps(
                    {
                        "format": 1,
                        "createdAt": time.strftime(
                            "%Y-%m-%dT%H:%M:%S%z"
                        ),
                        "rekordboxXml": str(self.collection.xml),
                        "playlist": self.playlist.path,
                        "tracks": [
                            item.as_manifest_entry()
                            for item in results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            self._update(
                state="done",
                stage="Finished",
                current="",
                progress=100,
                completed=len(tracks),
                position=len(tracks),
                manifest=manifest,
                eta_seconds=0.0,
                elapsed_seconds=time.monotonic() - self._started,
            )

        except Cancelled:
            self._update(
                state="cancelled",
                stage="Cancelled",
                current="",
            )

        except Exception as error:
            self._update(
                state="failed",
                stage="Error",
                current="",
                fatal=f"{type(error).__name__}: {error}",
            )

        return self.state