#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
import whisper
from pyannote.audio import Pipeline

# Load environment variables (e.g., PYANNOTE_KEY) from .env
load_dotenv()

# Determine preferred device for Whisper (MPS or CUDA), fallback to CPU
if torch.backends.mps.is_available():
    whisper_device = 'mps'
elif torch.cuda.is_available():
    whisper_device = 'cuda'
else:
    whisper_device = 'cpu'
print(f"🛠 Whisper device: {whisper_device}")


def convert_to_wav(input_path: Path) -> Path:
    """Convert any media file to 16 kHz mono WAV."""
    wav_path = input_path.with_suffix('.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(input_path), '-ar', '16000', '-ac', '1', str(wav_path)],
        check=True
    )
    return wav_path


def parse_rttm(rttm_path: Path):
    """Parse RTTM file into list of (start, end, label)."""
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                segments.append((start, start + duration, label))
    return segments


def assign_speaker(segment: dict, diarization: list) -> str:
    """Assign speaker to a transcription segment by midpoint timestamp."""
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

    # 2. Load and run Whisper
    try:
        print(f" Loading Whisper model ({model_size}) on {whisper_device}...")
        model = whisper.load_model(model_size, device=whisper_device)
    except NotImplementedError as e:
        print(f" Whisper on {whisper_device} failed: {e}. Using CPU.")
        model = whisper.load_model(model_size, device='cpu')
    print(f"⏳ Transcribing {wav.name}...")
    result = model.transcribe(str(wav))

    # 3. Speaker diarization with PyAnnote (CPU)
    hf_key = os.getenv('PYANNOTE_KEY')
    if not hf_key:
        raise RuntimeError('Missing PYANNOTE_KEY in environment')
    print(f" Initializing PyAnnote speaker-diarization pipeline...")
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=hf_key)
    print(f" Running diarization on {wav.name}...")
    diarization_result = pipeline(str(wav))

    # 4. Write RTTM and parse
    rttm_path = wav.with_suffix('.rttm')
    with open(rttm_path, 'w') as f:
        f.write(diarization_result.to_rttm())
    diarization = parse_rttm(rttm_path)

    # 5. Merge and write speaker-labeled transcript
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for seg in result.get('segments', []):
            speaker = assign_speaker(seg, diarization)
            text = seg['text'].strip()
            fout.write(f"{speaker}: {text}\n")
    print(f"saved speaker-labeled transcript to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe media and label speakers using Whisper + PyAnnote'
    )
    parser.add_argument('input', type=Path, help='Input media file')
    parser.add_argument('output', type=Path, help='Output transcript (.txt)')
    parser.add_argument('--model', type=str, default='base', help='Whisper model size')
    args = parser.parse_args()
    transcribe_speakers(args.input, args.output, args.model)
