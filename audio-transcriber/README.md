# Audio Transcriber

A fully local Zoom-call transcription tool using OpenAI Whisper.  
No cloud uploads, no per-minute fees.

## Features

- **Local** transcription of long videos (30–60 min)  
- Whisper model sizes: `tiny` → `large`  
- **Optional** speaker-diarization via PyAnnote

## File Structure

audio-transcriber/
├── environment.yml
├── README.md
├── .gitignore
└── src/
    ├── transcribe.py # main transcription
    └── diarize.py # optional speaker-diarization

## Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# 2. Create & activate conda env
conda env create --name audio-transcriber --file=environment.yml
conda activate audio-transcriber
```

