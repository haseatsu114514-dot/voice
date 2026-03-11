from __future__ import annotations

import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "audio path is required"}))
        return 1

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(json.dumps({"error": f"file not found: {audio_path}"}))
        return 1

    model_name = "small"
    model = WhisperModel(model_name, device="auto", compute_type="int8_float16")
    segments, _ = model.transcribe(
        str(audio_path),
        language="ja",
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = "".join(segment.text for segment in segments).strip()
    print(json.dumps({"text": text}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
