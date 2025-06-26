from pathlib import Path
from pyannote.audio import Pipeline
import argparse

def diarize(audio_path: Path, out_rttm: Path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(str(audio_path))
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rttm, "w") as f:
        f.write(diarization.to_rttm())
    print(f"Speaker-diarization saved to `{out_rttm}`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run speaker diarization on a WAV file"
    )
    parser.add_argument("audio", type=Path, help="Path to .wav file")
    parser.add_argument(
        "rttm", type=Path, help="Path to write RTTM speaker segments"
    )
    args = parser.parse_args()
    diarize(args.audio, args.rttm)
