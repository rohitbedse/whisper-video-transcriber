
from faster_whisper import WhisperModel

video_path = "v1.mp4"
output_file = "v1_transcript.txt"

model = WhisperModel("base", device="cpu", compute_type="int8")

print("Transcribing:", video_path)

segments, info = model.transcribe(video_path)

with open(output_file, "w", encoding="utf-8") as f:
    for segment in segments:
        start = segment.start
        end = segment.end
        text = segment.text.strip()

        line = f"[{start:.2f}s - {end:.2f}s] {text}\n"
        f.write(line)
        print(line, end="")

print("Saved:", output_file)
