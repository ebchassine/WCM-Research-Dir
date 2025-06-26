import whisper
import argparse
import subprocess
from pathlib import Path

def convert_video_to_audio(video_path: Path) -> Path:
    audio_path = video_path.with_suffix('.wav')
    # extract 16 kHz single-channel WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", str(audio_path)
    ], check=True)
    return audio_path

def transcribe(video_path: Path, out_txt: Path, model_size: str):
    model = whisper.load_model(model_size)
    audio_path = convert_video_to_audio(video_path)
    print(f"⏳ Transcribing `{audio_path.name}` with Whisper `{model_size}`…")
    result = model.transcribe(str(audio_path))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(result["text"], encoding="utf-8")
    print(f"✅ Saved transcript to `{out_txt}`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe Zoom video locally using Whisper"
    )
    parser.add_argument("input", type=Path, help="Path to Zoom video file")
    parser.add_argument("output", type=Path, help="Path to write transcript (.txt)")
    parser.add_argument(
        "--model", type=str, default="base",
        help="Whisper model size: tiny, base, small, medium, large"
    )
    args = parser.parse_args()
    transcribe(args.input, args.output, args.model)
