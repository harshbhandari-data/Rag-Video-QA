from pathlib import Path
import json
from cuda_dll_path import bootstrap_cuda_dll_path

bootstrap_cuda_dll_path()

from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audios"
OUTPUT_DIR = PROJECT_ROOT / "data" / "transcripts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Whisper model...")

model = WhisperModel(
    "large-v3",              #medium has less accuracy but process vedio faster
    device="cuda",
    compute_type="float16"
)

print("Model loaded!\n")

# to test
# audio_files = list(AUDIO_DIR.glob("test.mp3"))

audio_files = list(AUDIO_DIR.glob("*.mp3"))

if not audio_files:
    print("No audio files found.")
    exit()

for audio_file in audio_files:

    output_file = OUTPUT_DIR / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"Skipping {audio_file.name} (already transcribed)")
        continue

    print(f"Converting: {audio_file.name}")
    
    #for medium whisper model use commneted fields to get better result

    segments, info = model.transcribe(
        str(audio_file),
        beam_size=5,  #10
        task="transcribe",
        language="hi",
        #best_of=5,
        #patience=2,
        condition_on_previous_text=True,
        vad_filter=True,  #false
        initial_prompt=
                        """
                        This is a machine learning lecture in Hindi and English.
                        The speaker frequently uses technical terms like Python, Machine Learning,
                        Deep Learning, NumPy, Pandas, TensorFlow, Scikit-learn, Regression,
                        Classification, Neural Networks, Dataset and Model.
                        """
    )

    transcript = {
        "lecture_name": audio_file.stem,
        "audio_file": audio_file.name,
        "language": info.language,
        "duration": round(info.duration, 2),
        "segments": []
    }

    for segment in segments:
        transcript["segments"].append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            transcript,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Saved: {output_file.name}\n")

print("All files converted successfully!")