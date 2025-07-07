#!/usr/bin/env python3
"""
Segment a video/audio into fixed-duration blocks, transcribe each block asynchronously with Whisper,
then concatenate and merge transcripts to avoid mid-sentence cuts.

Usage:
python src/single_speaker.py \
    --input video-sources/video1.mp4 \
    --output /results/sinle_results.txt \
    --model base \
    --block-duration 600 \
    --overlap 5 \
    --merge-threshold 0.5

python src/single_speaker.py \
  --input video-sources/video1.mp4 \
  --output output/single_results.txt \
  --model base \
  --block-duration 600 \
  --overlap 5 \
  --merge-threshold 0.5

"""
import argparse
import asyncio
import subprocess
import os
import tempfile
import shutil
from pathlib import Path

import soundfile as sf
import whisper


def convert_to_wav(input_path: Path) -> Path:
    """Convert video to mono 16kHz WAV if needed."""
    if input_path.suffix.lower() == '.wav':
        return input_path
    wav_path = input_path.with_suffix('.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(input_path), '-ar', '16000', '-ac', '1', str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return wav_path


def get_duration(wav_path: Path) -> float:
    """Return duration of WAV file in seconds."""
    with sf.SoundFile(str(wav_path)) as f:
        return len(f) / f.samplerate


def transcribe_block_sync(wav_path: Path, start: float, end: float,
                           model_size: str, temp_dir: Path, idx: int):
    """Extract a block of audio and transcribe it, returning timestamped segments."""
    # Prepare block file
    block_file = temp_dir / f"block_{idx}.wav"
    # Extract segment with ffmpeg
    subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(wav_path),
            '-ss', str(start), '-to', str(end),
            '-ar', '16000', '-ac', '1', str(block_file)
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Load model and transcribe
    model = whisper.load_model(model_size)
    result = model.transcribe(str(block_file), word_timestamps=False)
    # Cleanup block file
    try:
        block_file.unlink()
    except OSError:
        pass
    # Adjust segment timestamps and collect
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start': seg['start'] + start,
            'end': seg['end'] + start,
            'text': seg['text'].strip()
        })
    return segments


def parse_args():
    p = argparse.ArgumentParser(
        description="Asynchronously segment and transcribe single-speaker audio."
    )
    p.add_argument('--input',   type=Path, required=True, help='Input video or WAV file')
    p.add_argument('--output',  type=Path, required=True, help='Output transcript (.txt)')
    p.add_argument('--model',   type=str,  default='base', help='Whisper model size')
    p.add_argument('--block-duration', type=float, default=600,
                   help='Block duration in seconds (default: 600)')
    p.add_argument('--overlap', type=float, default=5,
                   help='Overlap between blocks in seconds (default: 5)')
    p.add_argument('--merge-threshold', type=float, default=0.5,
                   help='Merge gap threshold in seconds (default: 0.5)')
    return p.parse_args()


async def main():
    args = parse_args()
    # Prepare audio
    wav = convert_to_wav(args.input)
    total_dur = get_duration(wav)
    # Compute segments
    specs = []  # list of (start, end, idx)
    idx = 0
    start = 0.0
    while start < total_dur:
        end = min(start + args.block_duration, total_dur)
        seg_start = start if idx == 0 else max(0.0, start - args.overlap)
        seg_end = min(end + args.overlap, total_dur)
        specs.append((seg_start, seg_end, idx))
        start += args.block_duration
        idx += 1
    # Temporary directory for blocks
    temp_dir = Path(tempfile.mkdtemp())
    # Launch async transcription tasks
    tasks = [
        asyncio.to_thread(
            transcribe_block_sync,
            wav, seg_start, seg_end,
            args.model, temp_dir, idx
        )
        for seg_start, seg_end, idx in specs
    ]
    results = await asyncio.gather(*tasks)
    # Flatten and sort
    all_segs = [s for block in results for s in block]
    all_segs.sort(key=lambda x: x['start'])
    # Merge close segments
    merged = []
    for seg in all_segs:
        if merged and seg['start'] <= merged[-1]['end'] + args.merge_threshold:
            merged[-1]['text'] += ' ' + seg['text']
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg.copy())
    # Write transcript
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transcript = '\n'.join(m['text'] for m in merged)
    args.output.write_text(transcript, encoding='utf-8')
    print(f"Saved transcript to {args.output}")
    # Cleanup
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    asyncio.run(main())
