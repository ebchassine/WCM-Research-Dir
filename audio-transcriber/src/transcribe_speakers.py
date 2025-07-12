#!/usr/bin/env python

# Example usage:
"""
python src/transcribe_speakers.py \
    --input files/testing_files/test.mp4 \
    --output results/test_output.txt \
    --model large \
    --block-duration 600 \
    --overlap 5


python src/transcribe_speakers.py \
    --input files/testing_files/test.mp4 \
    --output results/test_output.txt \
    --model medium \
    --block-duration 600 \
    --overlap 5
"""

import os
import argparse
import tempfile
import subprocess
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

import whisper
from pyannote.audio import Pipeline

# Load environment variables (for PYANNOTE_API_KEY)
load_dotenv()
PYANNOTE_API_KEY = os.getenv("PYANNOTE_KEY")


def get_video_duration(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
         'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True
    )
    assert result.stdout.strip(), "get_video_duration failed"
    # print("LENGTH OUTPUT", result.stdout.strip())
    return float(result.stdout.strip())


def split_video(video_path, output_dir, segment_duration=600, overlap=5):
    os.makedirs(output_dir, exist_ok=True)
    total_duration = get_video_duration(video_path)
    segments = []
    start = 0
    i = 0
    while start < total_duration:
        end = min(start + segment_duration + overlap, total_duration)
        segment_file = f"{output_dir}/segment_{i:03d}.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start), '-to', str(end),
            '-c', 'copy', segment_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segments.append((segment_file, start, end))
        start += segment_duration
        i += 1
    return segments


def convert_to_wav(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-f", "wav", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def transcribe_segment(segment_path, model_name='base'):
    model = whisper.load_model(model_name)
    result = model.transcribe(segment_path, word_timestamps=True)
    return result


def diarize_segment(wav_path):
    if not PYANNOTE_API_KEY:
        raise EnvironmentError("Missing PYANNOTE_API_KEY in environment")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=PYANNOTE_API_KEY)
    diarization = pipeline(wav_path)
    return diarization


def process_segment(args):
    segment_path, start_time, _, model_name = args
    try:
        # Convert to WAV for diarization
        wav_path = segment_path.replace(".mp4", ".wav")
        convert_to_wav(segment_path, wav_path)

        # Transcribe (can use .mp4)
        transcription = transcribe_segment(segment_path, model_name)

        # Diarize (must use .wav)
        diarization = diarize_segment(wav_path)

        entries = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_text = ''
            for seg in transcription['segments']:
                seg_start = seg['start']
                seg_end = seg['end']
                abs_start = start_time + seg_start
                abs_end = start_time + seg_end
                if abs_start >= start_time + turn.start and abs_end <= start_time + turn.end:
                    segment_text += seg['text'].strip() + ' '
            if segment_text.strip():
                entries.append({
                    'start': round(start_time + turn.start, 2),
                    'end': round(start_time + turn.end, 2),
                    'speaker': speaker,
                    'text': segment_text.strip()
                })
        return entries
    except Exception as e:
        print(f"Error processing segment {segment_path}: {e}")
        return []


def run_pipeline(video_path, output_txt, model_name='base', block_duration=600, overlap=5):
    with tempfile.TemporaryDirectory() as tmpdir:
        segments = split_video(video_path, tmpdir, segment_duration=block_duration, overlap=overlap)
        print(f"Segmented video into {len(segments)} parts.")

        args = [(seg_path, start, end, model_name) for seg_path, start, end in segments]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_segment, args)

        all_entries = [entry for sublist in results for entry in sublist]
        all_entries.sort(key=lambda x: x['start'])

        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, 'w') as f:
            for row in all_entries:
                speaker_label = row['speaker'].replace("SPEAKER_", "Speaker ")
                f.write(f"{speaker_label}: {row['text']}\n")
        print(f"Saved diarized script to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to output TXT file')
    parser.add_argument('--model', type=str, default='medium', help='Whisper model to use (e.g., tiny, base, small, medium, large)')
    parser.add_argument('--block-duration', type=int, default=600, help='Segment duration in seconds (default: 600)')
    parser.add_argument('--overlap', type=int, default=5, help='Overlap between segments in seconds (default: 5)')
    args = parser.parse_args()

    run_pipeline(args.input, args.output, model_name=args.model, block_duration=args.block_duration, overlap=args.overlap)
