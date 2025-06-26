#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline

# Load environment variables (e.g., PYANNOTE_KEY) from .env
load_dotenv()


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any media file to 16 kHz mono WAV.
    """
    wav_path = input_path.with_suffix('.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(input_path), '-ar', '16000', '-ac', '1', str(wav_path)],
        check=True
    )
    return wav_path


def parse_rttm(rttm_path: Path):
    """
    Parse an RTTM file into a list of (start, end, speaker_label) tuples.
    """
    segments = []
    with rttm_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                segments.append((start, start + duration, label))
    return segments


def assign_speaker(segment: dict, diarization: list) -> str:
    """
    Assign a speaker label to a transcription segment by its midpoint.
    """
    midpoint = (segment['start'] + segment['end']) / 2
    for start, end, label in diarization:
        if start <= midpoint < end:
            if '_' in label:
                try:
                    idx = int(label.split('_')[-1]) + 1
                    return f"Speaker {idx}"
                except ValueError:
                    pass
            return label
    return 'Speaker 1'


def transcribe_speakers(input_file: Path, output_file: Path, model_size: str):
    # 1. Convert to WAV
    wav = convert_to_wav(input_file)

    # 2. Transcribe with Whisper
    print(f"⏳ Transcribing {wav.name} with Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(str(wav))  # contains 'segments'

    # 3. Speaker diarization with PyAnnote
    hf_key = os.getenv('PYANNOTE_KEY')
    if not hf_key:
        raise RuntimeError('Missing PYANNOTE_KEY in environment for diarization')
    print(f"🗣️ Running speaker diarization on {wav.name}...")
    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization',
        use_auth_token=hf_key
    )
    diarization_result = pipeline(str(wav))

    # Write RTTM to file
    rttm_path = wav.with_suffix('.rttm')
    with rttm_path.open('w') as f:
        f.write(diarization_result.to_rttm())

    # Parse diarization segments
    diarization = parse_rttm(rttm_path)

    # 4. Merge transcription segments with speaker labels
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as fout:
        for seg in result.get('segments', []):
            speaker = assign_speaker(seg, diarization)
            text = seg['text'].strip()
            fout.write(f"{speaker}: {text}\n")

    print(f"✅ Speaker-labeled transcript saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe media file and label speakers using Whisper + PyAnnote'
    )
    parser.add_argument('input', type=Path, help='Path to input media file')
    parser.add_argument('output', type=Path, help='Path to output transcript (.txt)')
    parser.add_argument(
        '--model', type=str, default='base',
        help='Whisper model size: tiny | base | small | medium | large'
    )
    args = parser.parse_args()
    transcribe_speakers(args.input, args.output, args.model)
