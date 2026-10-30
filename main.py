
from faster_whisper import WhisperModel

video_path = "v1.mp4"
output_file = "v1_transcript1.txt"

# CPU-friendly model
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Transcribing:", video_path)

segments, info = model.transcribe(
    video_path,
    language="en",       # Use this if the audio is in English
    beam_size=5,
    vad_filter=True
)

with open(output_file, "w", encoding="utf-8") as f:
    for segment in segments:
        start = segment.start
        end = segment.end
        text = segment.text.strip()

        line = f"[{start:.2f}s - {end:.2f}s] {text}\n"
        f.write(line)
        print(line, end="")

print("\nLanguage detected:", info.language)
print("Saved transcript:", output_file)
