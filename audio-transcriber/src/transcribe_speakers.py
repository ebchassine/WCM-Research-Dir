#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
import whisper
from pyannote.audio import Pipeline
"""

python src/transcribe_speakers.py \
    video-sources/test_1.mp4 \
    /results/script_results.txt \
    --model base

"""
load_dotenv()

# get device for Whisper
if torch.backends.mps.is_available():
    whisper_device = 'cuda' if False else 'cpu'  # option to fill in later when MPS is supported 
elif torch.cuda.is_available():
    whisper_device = 'cuda'
else:
    whisper_device = 'cpu'
print(f"🛠 Whisper device: {whisper_device}")


def convert_to_wav(input_path: Path) -> Path:
    wav_path = input_path.with_suffix('.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(input_path), '-ar', '16000', '-ac', '1', str(wav_path)],
        check=True
    )
    return wav_path


def parse_rttm(rttm_path: Path):
    segments = []
    with rttm_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                segments.append((start, start + duration, label))
    return segments


def assign_label(segment, diarization):
    # Return the original diarization label for segment midpoint
    midpoint = (segment['start'] + segment['end']) / 2
    for start, end, label in diarization:
        if start <= midpoint < end:
            return label
    return None


def transcribe_speakers(input_file: Path, output_file: Path, model_size: str):
    # Convert to WAV
    wav = convert_to_wav(input_file)

    # Whisper transcription
    print(f"Loading Whisper model ({model_size}) on {whisper_device}...")
    model = whisper.load_model(model_size, device=whisper_device)
    print(f"Transcribing {wav.name}...")
    result = model.transcribe(str(wav), word_timestamps=True)

    # Diarization 
    hf_key = os.getenv('PYANNOTE_KEY')
    if not hf_key:
        raise RuntimeError('PYANNOTE_KEY not set')
    print(f"Init PyAnnote pipeline...")
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=hf_key)
    print(f"Running diarization on {wav.name}...")
    diar_res = pipeline(str(wav))
    rttm_path = wav.with_suffix('.rttm')
    with rttm_path.open('w') as f:
        f.write(diar_res.to_rttm())
    diarization = parse_rttm(rttm_path)

    # Map raw labels to speaker k
    label_map = {}
    order = sorted({label for _, _, label in diarization}, key=lambda l: next(s for s in diarization if s[2]==l)[0])
    for idx, lab in enumerate(order, start=1):
        label_map[lab] = f"Speaker {idx}"

    # Merge segments by speaker and time gap
    merged = []
    last_speaker = None
    last_end = 0.0
    for seg in result['segments']:
        lab = assign_label(seg, diarization)
        speaker = label_map.get(lab, 'Speaker 1')
        text = seg['text'].strip()
        if speaker == last_speaker and seg['start'] - last_end < 0.5:
            merged[-1] = merged[-1].rstrip() + ' ' + text
        else:
            merged.append(f"{speaker}: {text}")
        last_speaker = speaker
        last_end = seg['end']

    # Output write here 
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(merged), encoding='utf-8')
    print(f"Saved improved transcript to {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--model', default='base')
    args = p.parse_args()
    transcribe_speakers(args.input, args.output, args.model)
