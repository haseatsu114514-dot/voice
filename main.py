from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard as pynput_keyboard

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_HOTKEY = "<cmd>+<shift>+space" if sys.platform == "darwin" else "<ctrl>+<alt>+space"


@dataclass
class AppConfig:
    hotkey: str = DEFAULT_HOTKEY
    sample_rate: int = 16000
    input_device: int | str | None = None
    language: str | None = "ja"
    model_name: str = "large-v3"
    model_device: str = "auto"
    compute_type: str = "int8_float16"
    beam_size: int = 5
    vad_filter: bool = True
    paste_after_transcribe: bool = True
    initial_prompt: str | None = None
    use_common_replacements: bool = True
    common_replacements_file: str = "common_replacements_ja.toml"


@dataclass
class NormalizationConfig:
    remove_fillers: bool = True
    fillers: list[str] = field(
        default_factory=lambda: ["えーと", "えっと", "あの", "その", "うーん", "えー"]
    )
    replacements: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def _parse_input_device(value: Any) -> int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and not value.isdigit():
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError("`app.input_device` must be an integer index or empty.")


def _load_replacements_from_file(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    replacements = data.get("replacements", {})
    if not isinstance(replacements, dict):
        raise ValueError(f"`replacements` table is invalid in {path}")
    return {str(before): str(after) for before, after in replacements.items()}


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    app_data = data.get("app", {})
    norm_data = data.get("normalization", {})

    app = AppConfig(
        hotkey=app_data.get("hotkey", AppConfig.hotkey),
        sample_rate=int(app_data.get("sample_rate", AppConfig.sample_rate)),
        input_device=_parse_input_device(app_data.get("input_device", AppConfig.input_device)),
        language=app_data.get("language", AppConfig.language),
        model_name=app_data.get("model_name", AppConfig.model_name),
        model_device=app_data.get("model_device", AppConfig.model_device),
        compute_type=app_data.get("compute_type", AppConfig.compute_type),
        beam_size=int(app_data.get("beam_size", AppConfig.beam_size)),
        vad_filter=bool(app_data.get("vad_filter", AppConfig.vad_filter)),
        paste_after_transcribe=bool(
            app_data.get("paste_after_transcribe", AppConfig.paste_after_transcribe)
        ),
        initial_prompt=app_data.get("initial_prompt", AppConfig.initial_prompt),
        use_common_replacements=bool(
            app_data.get("use_common_replacements", AppConfig.use_common_replacements)
        ),
        common_replacements_file=str(
            app_data.get("common_replacements_file", AppConfig.common_replacements_file)
        ),
    )

    common_replacements: dict[str, str] = {}
    if app.use_common_replacements:
        common_replacements_path = Path(app.common_replacements_file)
        if not common_replacements_path.is_absolute():
            common_replacements_path = path.parent / common_replacements_path

        if common_replacements_path.exists():
            common_replacements = _load_replacements_from_file(common_replacements_path)
        else:
            print(f"[WARN] Common replacements file not found: {common_replacements_path}", flush=True)

    custom_replacements = {
        str(before): str(after) for before, after in dict(norm_data.get("replacements", {})).items()
    }

    norm = NormalizationConfig(
        remove_fillers=bool(norm_data.get("remove_fillers", True)),
        fillers=list(norm_data.get("fillers", [])) or NormalizationConfig().fillers,
        replacements={**common_replacements, **custom_replacements},
    )
    return Config(app=app, normalization=norm)


class AudioRecorder:
    def __init__(self, sample_rate: int, input_device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.input_device = input_device
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"[WARN] audio status: {status}", flush=True)
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.input_device,
            callback=self._callback,
            blocksize=0,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.array([], dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)
            self._frames = []
        return audio


class WhisperTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = WhisperModel(
            config.model_name,
            device=config.model_device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()


class TextNormalizer:
    def __init__(self, config: NormalizationConfig) -> None:
        self.remove_fillers = config.remove_fillers
        self.fillers = sorted(config.fillers, key=len, reverse=True)
        self.replacements = dict(sorted(config.replacements.items(), key=lambda item: len(item[0]), reverse=True))

    def normalize(self, text: str) -> str:
        normalized = text

        for before, after in self.replacements.items():
            normalized = normalized.replace(before, after)

        if self.remove_fillers:
            for filler in self.fillers:
                normalized = normalized.replace(filler, "")

        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"([、。！？])\1+", r"\1", normalized)
        return normalized.strip()


class OutputInserter:
    def __init__(self, paste_after_transcribe: bool) -> None:
        self.paste_after_transcribe = paste_after_transcribe
        self.keyboard = pynput_keyboard.Controller()

    def insert(self, text: str) -> None:
        if not text:
            print("[INFO] Empty result. Skipped insert.", flush=True)
            return
        pyperclip.copy(text)

        if not self.paste_after_transcribe:
            print(f"[COPIED] {text}", flush=True)
            return

        modifier = pynput_keyboard.Key.cmd if sys.platform == "darwin" else pynput_keyboard.Key.ctrl
        with self.keyboard.pressed(modifier):
            self.keyboard.press("v")
            self.keyboard.release("v")
        print(f"[PASTED] {text}", flush=True)


class VoiceInputApp:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.recorder = AudioRecorder(config.app.sample_rate, config.app.input_device)
        self.transcriber = WhisperTranscriber(config.app)
        self.normalizer = TextNormalizer(config.normalization)
        self.inserter = OutputInserter(config.app.paste_after_transcribe)
        self._recording = False
        self._state_lock = threading.Lock()

        self.hotkey_listener = pynput_keyboard.GlobalHotKeys(
            {
                self.config.app.hotkey: self._toggle_recording,
            }
        )

    def _toggle_recording(self) -> None:
        with self._state_lock:
            if self._recording:
                audio = self.recorder.stop()
                self._recording = False
                print("[INFO] Recording stopped. Transcribing...", flush=True)
                threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()
                return

            self.recorder.start()
            self._recording = True
            print("[INFO] Recording started.", flush=True)

    def _process_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            print("[WARN] No audio captured.", flush=True)
            return
        try:
            raw_text = self.transcriber.transcribe(audio)
            normalized = self.normalizer.normalize(raw_text)
            self.inserter.insert(normalized)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", flush=True)

    def run(self) -> None:
        print("=== PC Voice Input ===", flush=True)
        print(f"Hotkey: {self.config.app.hotkey} (start/stop)", flush=True)
        print(f"Replacements loaded: {len(self.normalizer.replacements)}", flush=True)
        print("Press Ctrl+C to exit.", flush=True)
        self.hotkey_listener.start()
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...", flush=True)
        finally:
            with self._state_lock:
                if self._recording:
                    self.recorder.stop()
                    self._recording = False
            self.hotkey_listener.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotkey-driven voice input tool")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to TOML config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    app = VoiceInputApp(config)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
