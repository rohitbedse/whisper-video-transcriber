import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path

import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────
MEDIA_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
MODEL_SIZE = "medium"
LANGUAGE = "hi"
BEAM_SIZE = 5
VAD_FILTER = True
BATCH_SIZE = 16
COMPUTE_TYPE = "float16" if shutil.which("nvidia-smi") else "int8"
DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"

INITIAL_PROMPT = (
    "यह एक हिंदी और अंग्रेजी मिश्रित वार्तालाप है। "
    "This is a mixed Hindi and English conversation. "
    "Speakers switch languages mid-sentence. "
    "Transcribe accurately in the spoken language."
)

OUTPUT_DIR = Path("transcription_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE} | Precision: {COMPUTE_TYPE} | Model: {MODEL_SIZE}")
print(f"Input: {MEDIA_PATH}")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────
print(f"\nLoading Whisper {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
t0 = time.time()
from faster_whisper import WhisperModel
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"   Loaded in {time.time() - t0:.1f}s")

# ─── AUDIO EXTRACTION ─────────────────────────────────────────────────────
def extract_audio(input_path: str, output_path: str = None) -> str:
    """Extract audio using ffmpeg, returns path to wav file."""
    if output_path is None:
        output_path = str(OUTPUT_DIR / "temp_audio.wav")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: winget install ffmpeg  OR  choco install ffmpeg")

    cmd = [
        "ffmpeg", "-nostdin", "-y", "-threads", "0", "-i", input_path,
        "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
    ]
    print(f"\nExtracting audio...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    print(f"   Done in {time.time()-t0:.1f}s")
    return output_path

# ─── TRANSCRIBE ───────────────────────────────────────────────────────────
audio_path = extract_audio(MEDIA_PATH)

print(f"\nTranscribing (beam={BEAM_SIZE}, batch={BATCH_SIZE})...\n")
t0 = time.time()

segments_iter, info = model.transcribe(
    audio_path,
    language=LANGUAGE,
    task="transcribe",
    beam_size=BEAM_SIZE,
    best_of=5,
    patience=1.0,
    vad_filter=VAD_FILTER,
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400,
        threshold=0.35,
    ),
    word_timestamps=True,
    condition_on_previous_text=True,
    temperature=0.0,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    initial_prompt=INITIAL_PROMPT,
    prepend_punctuations="\"'“¿([{-",
    append_punctuations="\"'.।,!?)]}،",
)

# ─── COLLECT RESULTS ──────────────────────────────────────────────────────
segments = []
for seg in segments_iter:
    segments.append({
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text.strip(),
        "avg_logprob": float(seg.avg_logprob),
        "no_speech_prob": float(seg.no_speech_prob),
        "words": [
            {"start": float(w.start), "end": float(w.end), "word": w.word, "probability": float(w.probability)}
            for w in (seg.words or [])
        ],
    })
    m, s = divmod(int(seg.end), 60)
    lang_tag = getattr(seg, 'language', '??')
    print(f"  [{m:02d}:{s:02d}] [{lang_tag}] {seg.text.strip()[:160]}")

elapsed = time.time() - t0
audio_dur = segments[-1]["end"] if segments else 0
speed = audio_dur / elapsed if elapsed else 0

# ─── SAVE OUTPUTS ─────────────────────────────────────────────────────────
stem = Path(MEDIA_PATH).stem
txt_path = OUTPUT_DIR / f"{stem}_transcript.txt"
vtt_path = OUTPUT_DIR / f"{stem}_transcript.vtt"
json_path = OUTPUT_DIR / f"{stem}_transcript.json"

def vtt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

with txt_path.open("w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"[{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}] {seg['text']}\n")

with vtt_path.open("w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for i, seg in enumerate(segments, 1):
        f.write(f"{i}\n{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}\n{seg['text']}\n\n")

json_path.write_text(json.dumps({
    "model": MODEL_SIZE,
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": segments,
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n{'─'*50}")
print(f"Done! {len(segments)} segs | {audio_dur/60:.1f} min audio | {elapsed/60:.1f} min proc | {speed:.1f}x RT")
print(f"   Detected: {info.language} ({info.language_probability:.0%})")
print(f"   Saved -> {txt_path}")
print(f"   Saved -> {vtt_path}")
print(f"   Saved -> {json_path}")

# Cleanup temp audio
try:
    Path(audio_path).unlink(missing_ok=True)
except:
    pass