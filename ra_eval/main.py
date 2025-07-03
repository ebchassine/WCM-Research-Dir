import os
import subprocess
import pandas as pd
import tempfile
import argparse
from pyannote.audio import Pipeline as DiarizationPipeline
import whisper
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv


def extract_audio(video_path: str, audio_path: str):
    """
    Extracts mono, 16kHz WAV audio from a video file using ffmpeg.
    """
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", audio_path
    ]
    subprocess.run(command, check=True)


def run_diarization(diar_pipeline: DiarizationPipeline, audio_path: str):
    """
    Runs speaker diarization and returns list of segments with start, end, speaker label.
    """
    diarization = diar_pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start_time': turn.start,
            'end_time': turn.end,
            'speaker': speaker
        })
    return segments


def run_transcription(whisper_model, audio_path: str, segments: list):
    """
    Transcribes each diarized segment by slicing audio via ffmpeg and running Whisper.
    Returns list of texts.
    """
    texts = []
    for idx, seg in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        # extract segment
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(seg['start_time']),
            "-to", str(seg['end_time']),
            tmp_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # transcribe
        result = whisper_model.transcribe(tmp_path)
        texts.append(result['text'].strip())
        os.remove(tmp_path)
    return texts


def analyze_sentiment(sentiment_model, texts: list):
    """
    Runs sentiment analysis on each text segment.
    Returns list of dicts with label and score.
    """
    sentiments = []
    for text in texts:
        if not text:
            sentiments.append({'label': None, 'score': None})
            continue
        out = sentiment_model(text)[0]
        sentiments.append({'label': out['label'], 'score': out['score']})
    return sentiments


def build_dataframe(segments: list, texts: list, sentiments: list) -> pd.DataFrame:
    """
    Combines diarization, transcription, and sentiment into a pandas DataFrame.
    Calculates duration, latency, interruption, overlap.
    """
    rows = []
    prev_end = 0.0
    prev_speaker = None

    for idx, seg in enumerate(segments):
        start = seg['start_time']
        end = seg['end_time']
        speaker = seg['speaker']
        text = texts[idx]
        sentiment = sentiments[idx]

        duration = end - start
        latency = start - prev_end if idx > 0 else None
        interruption = (start < prev_end) and (speaker != prev_speaker) if idx > 0 else False
        overlap_with = []
        # simple overlap detection
        if interruption:
            overlap_with = [idx - 1]

        rows.append({
            'segment_id': idx,
            'speaker': speaker,
            'text': text,
            'start_time': start,
            'end_time': end,
            'duration': duration,
            'latency_from_prev': latency,
            'interruption': interruption,
            'overlap_with': overlap_with,
            'sentiment_label': sentiment['label'],
            'sentiment_score': sentiment['score']
        })

        prev_end = end
        prev_speaker = speaker

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Therapy session evaluation pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--pyannote_token", required=True, help="Huggingface token for pyannote.audio pipeline")
    args = parser.parse_args()

    # 1. Extract audio
    audio_path = "session_audio.wav"
    extract_audio(args.video, audio_path)

    # 2. Load models
    print("Loading models...")
    diar_pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=args.pyannote_token
    )
    whisper_model = whisper.load_model("base")
    sentiment_model = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # 3. Diarization
    print("Running diarization...")
    segments = run_diarization(diar_pipeline, audio_path)

    # 4. Transcription
    print("Running transcription...")
    texts = run_transcription(whisper_model, audio_path, segments)

    # 5. Sentiment
    print("Analyzing sentiment...")
    sentiments = analyze_sentiment(sentiment_model, texts)

    # 6. Build DataFrame and save
    print("Building DataFrame and saving to CSV...")
    df = build_dataframe(segments, texts, sentiments)
    df.to_csv(args.output, index=False)

    print(f"Pipeline complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
