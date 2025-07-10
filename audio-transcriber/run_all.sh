#!/bin/bash

# List of STAR IDs to process
star_ids=(270)

# Segments to check
segments=("eval" "bl" "wk3" "wk5" "wk9")

# Base directory for audio files
FILES_DIR="./files"
OUTPUT_DIR="./outputs"
# Paths to scripts
SINGLE_SCRIPT="./src/single_speaker.py"
DUAL_SCRIPT="./src/transcribe_speakers.py"
# Whisper model size
MODEL="large"

echo "Iterating through STAR IDS: ${star_ids}"
echo "Base dir: ${FILES_DIR}"

echo ""

for star_id in "${star_ids[@]}"; do
    
    star_dir="${FILES_DIR}/STAR${star_id}"
    echo "STAR${star_id}"
    # Skip if STAR directory doesn't exist
    [ -d "$star_dir" ] || echo "Skiped ${star_dir}" || continue

    for segment in "${segments[@]}"; do
        pattern="${star_dir}/STAR${star_id}.${segment}."*.m4a
        for filepath in $pattern; do
            # If no matching files, skip
            [ -e "$filepath" ] || continue

            filename=$(basename "$filepath")

            # Determine audio type
            if [[ "$filename" == *_audioex.m4a ]]; then
                speaker="audioex"
                script=$SINGLE_SCRIPT
            elif [[ "$filename" == *_audiopt.m4a ]]; then
                speaker="audiopt"
                script=$SINGLE_SCRIPT
            elif [[ "$filename" == *.m4a ]]; then
                speaker="full"
                script=$DUAL_SCRIPT
            else
                echo "Skipping unrecognized file: $filepath"
                continue
            fi

            output_dir="outputs/STAR${star_id}/${segment}"
            mkdir -p "$output_dir"
            output_file="${output_dir}/${speaker}_transcript.txt"

            echo "Processing $filepath → $output_file"

            python "$script" --input "$filepath" --output "$output_file" --model "$MODEL"
        done
    done
done
