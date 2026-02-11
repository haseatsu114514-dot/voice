from __future__ import annotations

import argparse
import difflib
import plistlib
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard as pynput_keyboard

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover
    tomli_w = None


DEFAULT_RECORD_HOTKEY = "<cmd>+<shift>+space" if sys.platform == "darwin" else "<ctrl>+<alt>+space"
DEFAULT_LEARN_HOTKEY = "<cmd>+<shift>+l" if sys.platform == "darwin" else "<ctrl>+<alt>+l"
LAUNCH_AGENT_LABEL = "com.haseatsu.voiceinput"


@dataclass
class AppConfig:
    hotkey: str = DEFAULT_RECORD_HOTKEY
    learn_hotkey: str = DEFAULT_LEARN_HOTKEY
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
    use_user_replacements: bool = True
    user_replacements_file: str = "user_replacements_ja.toml"
    auto_stop_silence_enabled: bool = True
    auto_stop_silence_sec: float = 1.2
    auto_stop_min_record_sec: float = 0.7
    silence_level_threshold: float = 0.01


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
    raise ValueError("`app.input_device` must be an integer index, device name, or empty.")


def _resolve_relative_path(base_config_path: Path, target: str) -> Path:
    path = Path(target)
    if path.is_absolute():
        return path
    return base_config_path.parent / path


def _load_replacements_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    replacements = data.get("replacements", {})
    if not isinstance(replacements, dict):
        raise ValueError(f"`replacements` table is invalid in {path}")
    return {str(before): str(after) for before, after in replacements.items()}


def _save_toml_data(data: dict[str, Any], path: Path) -> None:
    if tomli_w is None:
        raise RuntimeError("tomli-w is required to save settings. Please install dependencies again.")
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    app_data = data.get("app", {})
    norm_data = data.get("normalization", {})

    app = AppConfig(
        hotkey=app_data.get("hotkey", AppConfig.hotkey),
        learn_hotkey=app_data.get("learn_hotkey", AppConfig.learn_hotkey),
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
        use_user_replacements=bool(app_data.get("use_user_replacements", AppConfig.use_user_replacements)),
        user_replacements_file=str(app_data.get("user_replacements_file", AppConfig.user_replacements_file)),
        auto_stop_silence_enabled=bool(
            app_data.get("auto_stop_silence_enabled", AppConfig.auto_stop_silence_enabled)
        ),
        auto_stop_silence_sec=float(app_data.get("auto_stop_silence_sec", AppConfig.auto_stop_silence_sec)),
        auto_stop_min_record_sec=float(
            app_data.get("auto_stop_min_record_sec", AppConfig.auto_stop_min_record_sec)
        ),
        silence_level_threshold=float(
            app_data.get("silence_level_threshold", AppConfig.silence_level_threshold)
        ),
    )

    common_replacements: dict[str, str] = {}
    if app.use_common_replacements:
        common_path = _resolve_relative_path(path, app.common_replacements_file)
        common_replacements = _load_replacements_from_file(common_path)

    user_replacements: dict[str, str] = {}
    if app.use_user_replacements:
        user_path = _resolve_relative_path(path, app.user_replacements_file)
        user_replacements = _load_replacements_from_file(user_path)

    custom_replacements = {
        str(before): str(after) for before, after in dict(norm_data.get("replacements", {})).items()
    }

    norm = NormalizationConfig(
        remove_fillers=bool(norm_data.get("remove_fillers", True)),
        fillers=list(norm_data.get("fillers", [])) or NormalizationConfig().fillers,
        replacements={**common_replacements, **user_replacements, **custom_replacements},
    )
    return Config(app=app, normalization=norm)


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int,
        input_device: int | str | None = None,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.level_callback = level_callback
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"[WARN] audio status: {status}", flush=True)
        with self._lock:
            self._frames.append(indata.copy())

        if self.level_callback is not None:
            rms = float(np.sqrt(np.mean(np.square(indata))))
            self.level_callback(rms)

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
        self._lock = threading.Lock()
        self.replacements: dict[str, str] = {}
        self.update_replacements(config.replacements)

    def update_replacements(self, replacements: dict[str, str]) -> None:
        sorted_items = dict(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))
        with self._lock:
            self.replacements = sorted_items

    def normalize(self, text: str) -> str:
        normalized = text

        with self._lock:
            replacements = list(self.replacements.items())

        for before, after in replacements:
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


def _is_meaningful_text(value: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ン一-龥A-Za-z0-9]", value))


def extract_learning_pairs(before_text: str, after_text: str) -> dict[str, str]:
    matcher = difflib.SequenceMatcher(None, before_text, after_text)
    pairs: dict[str, str] = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue

        before_fragment = before_text[i1:i2].strip()
        after_fragment = after_text[j1:j2].strip()

        if not before_fragment or not after_fragment:
            continue
        if before_fragment == after_fragment:
            continue
        if len(before_fragment) < 2 or len(after_fragment) < 2:
            continue
        if len(before_fragment) > 20 or len(after_fragment) > 24:
            continue
        if not _is_meaningful_text(before_fragment) or not _is_meaningful_text(after_fragment):
            continue

        pairs[before_fragment] = after_fragment

    return pairs


class LaunchAgentManager:
    @staticmethod
    def plist_path() -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"

    @staticmethod
    def install(python_executable: Path, script_path: Path, config_path: Path) -> Path:
        plist_path = LaunchAgentManager.plist_path()
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        logs_dir = Path.home() / "Library" / "Logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "Label": LAUNCH_AGENT_LABEL,
            "ProgramArguments": [
                str(python_executable),
                str(script_path),
                "--config",
                str(config_path),
                "--tray",
            ],
            "RunAtLoad": True,
            "KeepAlive": True,
            "WorkingDirectory": str(script_path.parent),
            "StandardOutPath": str(logs_dir / "voice_input.out.log"),
            "StandardErrorPath": str(logs_dir / "voice_input.err.log"),
        }

        with plist_path.open("wb") as fp:
            plistlib.dump(payload, fp)

        subprocess.run(["launchctl", "unload", str(plist_path)], check=False, capture_output=True)
        loaded = subprocess.run(["launchctl", "load", str(plist_path)], check=False, text=True, capture_output=True)
        if loaded.returncode != 0:
            raise RuntimeError(loaded.stderr.strip() or "failed to load launch agent")

        return plist_path

    @staticmethod
    def uninstall() -> Path:
        plist_path = LaunchAgentManager.plist_path()
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False, capture_output=True)
        if plist_path.exists():
            plist_path.unlink()
        return plist_path


class VoiceInputEngine:
    def __init__(self, config: Config, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path

        self.transcriber = WhisperTranscriber(config.app)
        self.normalizer = TextNormalizer(config.normalization)
        self.inserter = OutputInserter(config.app.paste_after_transcribe)
        self.recorder = AudioRecorder(
            config.app.sample_rate,
            config.app.input_device,
            level_callback=self._on_audio_level,
        )

        self._recording = False
        self._record_started_at = 0.0
        self._last_voice_at = 0.0
        self._state_lock = threading.Lock()
        self._hotkeys_started = False

        self.last_output_text = ""
        self.last_raw_text = ""

        self.hotkey_listener = self._build_hotkey_listener()

    def _build_hotkey_listener(self) -> pynput_keyboard.GlobalHotKeys:
        hotkeys: dict[str, Callable[[], None]] = {
            self.config.app.hotkey: self._toggle_recording,
        }

        learn_hotkey = self.config.app.learn_hotkey.strip()
        if learn_hotkey:
            if learn_hotkey == self.config.app.hotkey:
                print("[WARN] `learn_hotkey` is same as `hotkey`; learning hotkey is disabled.", flush=True)
            else:
                hotkeys[learn_hotkey] = self.learn_from_clipboard

        return pynput_keyboard.GlobalHotKeys(hotkeys)

    def _on_audio_level(self, rms: float) -> None:
        if rms >= self.config.app.silence_level_threshold:
            with self._state_lock:
                self._last_voice_at = time.monotonic()

    def start_hotkeys(self) -> None:
        if self._hotkeys_started:
            return
        self.hotkey_listener.start()
        self._hotkeys_started = True

    def stop_hotkeys(self) -> None:
        if not self._hotkeys_started:
            return
        self.hotkey_listener.stop()
        self._hotkeys_started = False

    def _start_recording(self) -> None:
        self.recorder.start()
        now = time.monotonic()
        self._recording = True
        self._record_started_at = now
        self._last_voice_at = now
        print("[INFO] Recording started.", flush=True)

    def _stop_recording(self, reason: str) -> None:
        audio = self.recorder.stop()
        self._recording = False
        print(f"[INFO] Recording stopped ({reason}). Transcribing...", flush=True)
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def _toggle_recording(self) -> None:
        with self._state_lock:
            if self._recording:
                self._stop_recording("hotkey")
                return
            self._start_recording()

    def check_auto_stop(self) -> None:
        should_stop = False

        with self._state_lock:
            if not self._recording:
                return
            if not self.config.app.auto_stop_silence_enabled:
                return

            now = time.monotonic()
            if now - self._record_started_at < self.config.app.auto_stop_min_record_sec:
                return
            if now - self._last_voice_at >= self.config.app.auto_stop_silence_sec:
                should_stop = True

        if should_stop:
            with self._state_lock:
                if self._recording:
                    self._stop_recording("silence")

    def _process_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            print("[WARN] No audio captured.", flush=True)
            return

        try:
            raw_text = self.transcriber.transcribe(audio)
            normalized = self.normalizer.normalize(raw_text)

            self.last_raw_text = raw_text
            self.last_output_text = normalized

            self.inserter.insert(normalized)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", flush=True)

    def _user_replacements_path(self) -> Path:
        return _resolve_relative_path(self.config_path, self.config.app.user_replacements_file)

    def _reload_replacements_only(self) -> None:
        refreshed = load_config(self.config_path)
        self.normalizer.update_replacements(refreshed.normalization.replacements)

    def learn_from_clipboard(self) -> None:
        if not self.config.app.use_user_replacements:
            print("[WARN] User replacements are disabled in config.", flush=True)
            return

        source_text = self.last_output_text.strip()
        if not source_text:
            print("[WARN] No previous transcription found to learn from.", flush=True)
            return

        corrected_text = pyperclip.paste().strip()
        if not corrected_text:
            print("[WARN] Clipboard is empty. Copy corrected text first.", flush=True)
            return

        if corrected_text == source_text:
            print("[INFO] Clipboard text is same as last output. Nothing to learn.", flush=True)
            return

        learned_pairs = extract_learning_pairs(source_text, corrected_text)
        if not learned_pairs:
            print("[INFO] No reliable replacement pairs were detected.", flush=True)
            return

        replacements_path = self._user_replacements_path()
        replacements_path.parent.mkdir(parents=True, exist_ok=True)

        existing = _load_replacements_from_file(replacements_path)
        new_count = 0

        for before, after in learned_pairs.items():
            if existing.get(before) != after:
                existing[before] = after
                new_count += 1

        if new_count == 0:
            print("[INFO] Learned pairs already exist in user dictionary.", flush=True)
            return

        _save_toml_data({"replacements": existing}, replacements_path)
        self._reload_replacements_only()
        print(f"[LEARN] Added {new_count} replacements to {replacements_path}", flush=True)

    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recording

    def reload_full_config(self) -> None:
        with self._state_lock:
            if self._recording:
                raise RuntimeError("Stop recording before reloading config.")

        self.stop_hotkeys()
        refreshed = load_config(self.config_path)

        self.config = refreshed
        self.transcriber = WhisperTranscriber(refreshed.app)
        self.normalizer = TextNormalizer(refreshed.normalization)
        self.inserter = OutputInserter(refreshed.app.paste_after_transcribe)
        self.recorder = AudioRecorder(
            refreshed.app.sample_rate,
            refreshed.app.input_device,
            level_callback=self._on_audio_level,
        )
        self.hotkey_listener = self._build_hotkey_listener()
        self.start_hotkeys()

    def shutdown(self) -> None:
        with self._state_lock:
            if self._recording:
                self.recorder.stop()
                self._recording = False
        self.stop_hotkeys()


class VoiceInputCli:
    def __init__(self, engine: VoiceInputEngine) -> None:
        self.engine = engine

    def run(self) -> None:
        print("=== PC Voice Input ===", flush=True)
        print(f"Record hotkey: {self.engine.config.app.hotkey}", flush=True)
        print(f"Learn hotkey: {self.engine.config.app.learn_hotkey}", flush=True)
        print(
            f"Auto-stop(silence): {self.engine.config.app.auto_stop_silence_enabled} "
            f"({self.engine.config.app.auto_stop_silence_sec:.1f}s)",
            flush=True,
        )
        print(f"Replacements loaded: {len(self.engine.normalizer.replacements)}", flush=True)
        print("Press Ctrl+C to exit.", flush=True)

        self.engine.start_hotkeys()

        try:
            while True:
                time.sleep(0.2)
                self.engine.check_auto_stop()
        except KeyboardInterrupt:
            print("\n[INFO] Exiting...", flush=True)
        finally:
            self.engine.shutdown()


def _open_settings_window(config_path: Path) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
        from tkinter import ttk
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] tkinter is not available: {exc}", file=sys.stderr)
        return 1

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 1

    raw_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    app_data = raw_data.setdefault("app", {})
    norm_data = raw_data.setdefault("normalization", {})

    root = tk.Tk()
    root.title("Voice Input Settings")
    root.geometry("560x500")

    hotkey_var = tk.StringVar(value=str(app_data.get("hotkey", DEFAULT_RECORD_HOTKEY)))
    learn_hotkey_var = tk.StringVar(value=str(app_data.get("learn_hotkey", DEFAULT_LEARN_HOTKEY)))
    model_var = tk.StringVar(value=str(app_data.get("model_name", "large-v3")))
    beam_var = tk.StringVar(value=str(app_data.get("beam_size", 5)))

    auto_stop_enabled_var = tk.BooleanVar(
        value=bool(app_data.get("auto_stop_silence_enabled", AppConfig.auto_stop_silence_enabled))
    )
    auto_stop_sec_var = tk.StringVar(value=str(app_data.get("auto_stop_silence_sec", 1.2)))
    silence_threshold_var = tk.StringVar(value=str(app_data.get("silence_level_threshold", 0.01)))

    paste_var = tk.BooleanVar(value=bool(app_data.get("paste_after_transcribe", True)))
    use_common_var = tk.BooleanVar(value=bool(app_data.get("use_common_replacements", True)))
    use_user_var = tk.BooleanVar(value=bool(app_data.get("use_user_replacements", True)))

    fillers_var = tk.StringVar(
        value=", ".join(norm_data.get("fillers", ["えーと", "えっと", "あの", "その", "うーん", "えー"]))
    )

    row = 0

    def add_label_and_widget(label: str, widget: tk.Widget) -> None:
        nonlocal row
        ttk.Label(root, text=label).grid(row=row, column=0, padx=10, pady=6, sticky="w")
        widget.grid(row=row, column=1, padx=10, pady=6, sticky="ew")
        row += 1

    root.grid_columnconfigure(1, weight=1)

    add_label_and_widget("録音ホットキー", ttk.Entry(root, textvariable=hotkey_var))
    add_label_and_widget("学習ホットキー", ttk.Entry(root, textvariable=learn_hotkey_var))
    add_label_and_widget(
        "モデル",
        ttk.Combobox(
            root,
            textvariable=model_var,
            values=["small", "medium", "large-v3"],
            state="readonly",
        ),
    )
    add_label_and_widget("beam_size", ttk.Entry(root, textvariable=beam_var))
    add_label_and_widget("無音自動停止を使う", ttk.Checkbutton(root, variable=auto_stop_enabled_var))
    add_label_and_widget("無音停止までの秒数", ttk.Entry(root, textvariable=auto_stop_sec_var))
    add_label_and_widget("無音判定しきい値", ttk.Entry(root, textvariable=silence_threshold_var))
    add_label_and_widget("文字起こし後に自動貼り付け", ttk.Checkbutton(root, variable=paste_var))
    add_label_and_widget("共通辞書を使う", ttk.Checkbutton(root, variable=use_common_var))
    add_label_and_widget("学習辞書を使う", ttk.Checkbutton(root, variable=use_user_var))
    add_label_and_widget("フィラー一覧(カンマ区切り)", ttk.Entry(root, textvariable=fillers_var))

    def save_settings() -> None:
        try:
            app_data["hotkey"] = hotkey_var.get().strip() or DEFAULT_RECORD_HOTKEY
            app_data["learn_hotkey"] = learn_hotkey_var.get().strip() or DEFAULT_LEARN_HOTKEY
            app_data["model_name"] = model_var.get().strip() or "large-v3"
            app_data["beam_size"] = int(beam_var.get().strip())
            app_data["auto_stop_silence_enabled"] = bool(auto_stop_enabled_var.get())
            app_data["auto_stop_silence_sec"] = float(auto_stop_sec_var.get().strip())
            app_data["silence_level_threshold"] = float(silence_threshold_var.get().strip())
            app_data["paste_after_transcribe"] = bool(paste_var.get())
            app_data["use_common_replacements"] = bool(use_common_var.get())
            app_data["use_user_replacements"] = bool(use_user_var.get())

            fillers = [item.strip() for item in fillers_var.get().split(",") if item.strip()]
            if fillers:
                norm_data["fillers"] = fillers

            _save_toml_data(raw_data, config_path)
            messagebox.showinfo("保存完了", "設定を保存しました。\n実行中アプリは設定再読込が必要です。")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失敗", str(exc))

    def open_user_dict() -> None:
        user_file = str(app_data.get("user_replacements_file", "user_replacements_ja.toml"))
        user_path = _resolve_relative_path(config_path, user_file)
        user_path.parent.mkdir(parents=True, exist_ok=True)
        if not user_path.exists():
            _save_toml_data({"replacements": {}}, user_path)

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(user_path)], check=False)
            else:
                messagebox.showinfo("辞書ファイル", str(user_path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("エラー", str(exc))

    button_frame = ttk.Frame(root)
    button_frame.grid(row=row, column=0, columnspan=2, padx=10, pady=16, sticky="ew")

    ttk.Button(button_frame, text="保存", command=save_settings).pack(side="left", padx=4)
    ttk.Button(button_frame, text="学習辞書を開く", command=open_user_dict).pack(side="left", padx=4)
    ttk.Button(button_frame, text="閉じる", command=root.destroy).pack(side="right", padx=4)

    root.mainloop()
    return 0


def run_tray_app(engine: VoiceInputEngine, config_path: Path) -> int:
    if sys.platform != "darwin":
        print("[ERROR] Tray mode is currently implemented for macOS only.", file=sys.stderr)
        return 1

    try:
        import rumps
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to import rumps: {exc}", file=sys.stderr)
        return 1

    script_path = Path(__file__).resolve()

    class VoiceTrayApp(rumps.App):
        def __init__(self) -> None:
            super().__init__("Voice", quit_button=None)
            self.menu = [
                "録音開始/停止",
                "修正を学習(クリップボード)",
                "設定を開く",
                "設定を再読込",
                None,
                "ログイン起動を有効化",
                "ログイン起動を無効化",
                None,
                "終了",
            ]
            self.timer = rumps.Timer(self._on_tick, 0.4)

        def _on_tick(self, _sender: Any) -> None:
            engine.check_auto_stop()
            self.title = "Voice●" if engine.is_recording() else "Voice"

        @rumps.clicked("録音開始/停止")
        def on_toggle(self, _sender: Any) -> None:
            engine._toggle_recording()

        @rumps.clicked("修正を学習(クリップボード)")
        def on_learn(self, _sender: Any) -> None:
            try:
                engine.learn_from_clipboard()
                rumps.notification("Voice Input", "学習", "クリップボードから学習しました。")
            except Exception as exc:  # noqa: BLE001
                rumps.alert(f"学習エラー: {exc}")

        @rumps.clicked("設定を開く")
        def on_settings(self, _sender: Any) -> None:
            subprocess.Popen([
                str(Path(sys.executable)),
                str(script_path),
                "--config",
                str(config_path),
                "--settings",
            ])

        @rumps.clicked("設定を再読込")
        def on_reload(self, _sender: Any) -> None:
            try:
                engine.reload_full_config()
                rumps.notification("Voice Input", "再読込", "設定を再読込しました。")
            except Exception as exc:  # noqa: BLE001
                rumps.alert(f"再読込エラー: {exc}")

        @rumps.clicked("ログイン起動を有効化")
        def on_install_login(self, _sender: Any) -> None:
            try:
                plist_path = LaunchAgentManager.install(
                    python_executable=Path(sys.executable).resolve(),
                    script_path=script_path,
                    config_path=config_path,
                )
                rumps.notification("Voice Input", "ログイン起動", f"有効化しました: {plist_path}")
            except Exception as exc:  # noqa: BLE001
                rumps.alert(f"ログイン起動の有効化に失敗: {exc}")

        @rumps.clicked("ログイン起動を無効化")
        def on_uninstall_login(self, _sender: Any) -> None:
            try:
                plist_path = LaunchAgentManager.uninstall()
                rumps.notification("Voice Input", "ログイン起動", f"無効化しました: {plist_path}")
            except Exception as exc:  # noqa: BLE001
                rumps.alert(f"ログイン起動の無効化に失敗: {exc}")

        @rumps.clicked("終了")
        def on_quit(self, _sender: Any) -> None:
            self.timer.stop()
            engine.shutdown()
            rumps.quit_application()

        def run(self, **options: Any) -> None:  # type: ignore[override]
            engine.start_hotkeys()
            self.timer.start()
            super().run(**options)

    print("[INFO] Tray app started.", flush=True)
    print(f"[INFO] Record hotkey: {engine.config.app.hotkey}", flush=True)
    print(f"[INFO] Learn hotkey: {engine.config.app.learn_hotkey}", flush=True)

    app = VoiceTrayApp()
    app.run()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotkey-driven voice input tool")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to TOML config",
    )
    parser.add_argument("--tray", action="store_true", help="Run as a menu bar app (macOS)")
    parser.add_argument("--settings", action="store_true", help="Open simple settings GUI")
    parser.add_argument("--install-login-item", action="store_true", help="Enable auto start at login (macOS)")
    parser.add_argument("--uninstall-login-item", action="store_true", help="Disable auto start at login (macOS)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()

    if args.settings:
        return _open_settings_window(config_path)

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 1

    if args.install_login_item:
        if sys.platform != "darwin":
            print("[ERROR] Login item setup is for macOS only.", file=sys.stderr)
            return 1

        plist_path = LaunchAgentManager.install(
            python_executable=Path(sys.executable).resolve(),
            script_path=Path(__file__).resolve(),
            config_path=config_path,
        )
        print(f"[INFO] Login item installed: {plist_path}")
        return 0

    if args.uninstall_login_item:
        if sys.platform != "darwin":
            print("[ERROR] Login item setup is for macOS only.", file=sys.stderr)
            return 1

        plist_path = LaunchAgentManager.uninstall()
        print(f"[INFO] Login item uninstalled: {plist_path}")
        return 0

    config = load_config(config_path)
    engine = VoiceInputEngine(config=config, config_path=config_path)

    if args.tray:
        return run_tray_app(engine, config_path)

    cli = VoiceInputCli(engine)
    cli.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
