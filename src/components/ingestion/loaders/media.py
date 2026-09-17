from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


MEDIA_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpeg", ".mpg", ".m4v",
)


class MediaTranscriptionLoader(BaseLoader):
    name = "media-faster-whisper"
    extensions = MEDIA_EXTENSIONS

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        if not self.config.transcribe_audio_video:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise OptionalDependencyError(
                "Audio/video ingestion requires `faster-whisper`."
            ) from exc

        model = WhisperModel(
            self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
        )

        segments, info = model.transcribe(
            str(path),
            language=self.config.whisper_language,
            word_timestamps=self.config.whisper_word_timestamps,
            vad_filter=True,
        )

        for index, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            if not text:
                continue

            yield DocumentBlock(
                block_id="",
                text=text,
                block_type="transcript",
                source_locator=f"time:{segment.start:.2f}-{segment.end:.2f}",
                metadata={
                    "segment": index,
                    "start": segment.start,
                    "end": segment.end,
                    "detected_language": info.language,
                    "language_probability": info.language_probability,
                },
            )
