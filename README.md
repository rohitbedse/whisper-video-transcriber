# Video Transcription Tool

A simple Python script to transcribe video/audio files using Whisper (via faster-whisper) for efficient CPU-based transcription.

## Features

- **Fast transcription** using `faster-whisper` (CTranslate2 backend)
- **CPU optimized** with `int8` quantization
- **Automatic language detection** (defaults to English)
- **VAD filtering** to remove silence
- **Timestamped output** in text format

## Requirements

- Python 3.8+
- ffmpeg (must be in PATH)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (Windows)
winget install ffmpeg
# OR
choco install ffmpeg
```

## Usage

```bash
# Basic usage (transcribes v1.mp4 to v1_transcript1.txt)
python main.py

# Or specify custom input/output
# Edit main.py and change:
# video_path = "your_video.mp4"
# output_file = "your_transcript.txt"
```

## Configuration

Edit `main.py` to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `video_path` | Input video/audio file | `"v1.mp4"` |
| `output_file` | Output transcript file | `"v1_transcript1.txt"` |
| Model | Whisper model size | `"small"` |
| `device` | Compute device | `"cpu"` |
| `compute_type` | Precision | `"int8"` |
| `language` | Force language (None = auto) | `"en"` |
| `beam_size` | Beam search width | `5` |
| `vad_filter` | Voice activity detection | `True` |

## Model Options

| Model | Speed | Accuracy | VRAM (GPU) |
|-------|-------|----------|------------|
| `tiny` | Fastest | Low | ~1 GB |
| `base` | Fast | Good | ~1 GB |
| `small` | Balanced | Better | ~2 GB |
| `medium` | Slow | High | ~5 GB |
| `large-v3` | Slowest | Highest | ~10 GB |

For CPU: use `small` or `medium` with `compute_type="int8"`

## Output Format

```
[0.00s - 3.45s] Hello, this is a test transcription.
[3.45s - 7.12s] The video is being processed.
...
```

## Files

- `main.py` - Main transcription script
- `requirements.txt` - Python dependencies
- `.gitignore` - Ignores venv, cache, media files, outputs

## License

MIT