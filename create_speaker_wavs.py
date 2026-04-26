import json
import subprocess
from pathlib import Path

title = "Strait of Hormuz disruption threatens to shake global economy"

# Host-side paths for reading diarization JSON and creating output folder
diar_path = Path("pipeline_data/api/diarization") / f"{title}.json"
out_dir = Path("pipeline_data/speakers/es")
out_dir.mkdir(parents=True, exist_ok=True)

if not diar_path.exists():
    raise FileNotFoundError(f"Diarization file not found: {diar_path}")

data = json.loads(diar_path.read_text())
segments = data.get("segments", [])

seen = set()

for seg in segments:
    speaker = seg["speaker"]

    if speaker in seen:
        continue

    start = float(seg["start_s"])
    end = float(seg["end_s"])
    duration = max(1.0, min(end - start, 8.0))

    # Container-side paths, because ffmpeg will run inside foreign-whispers-api
    container_video_path = f"/app/pipeline_data/api/videos/{title}.mp4"
    container_out_path = f"/app/pipeline_data/speakers/es/{speaker}.wav"

    cmd = [
        "docker",
        "exec",
        "foreign-whispers-api",
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", container_video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        container_out_path,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Created pipeline_data/speakers/es/{speaker}.wav from {start:.2f}s to {start + duration:.2f}s")
    seen.add(speaker)

print("Done. Speakers:", sorted(seen))
