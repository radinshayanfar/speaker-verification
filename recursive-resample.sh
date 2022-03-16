#!/bin/bash

# A simple script to recursively resample a bunch of files
# in a directory. Only certain file extensions (mp3, aac,
# flac, wav) are considered.
#
# It takes 2 command line options: `indir` and `outdir`.
# The destination (`outdir`) is relative to the current
# directory of where you were when the script was run.
#
# Example: resample.sh audio/ resampled/
#
# The direcotry structure inside `indir` will be replicated
# in `outdir`.


# Sourece directory with files to convert
InDir=$1

# Set the directory you want for the converted files
OutDir=$2

# make sure the output directory exists (create it if not)
mkdir -p "$OutDir"

# Target sample rate
TARGET_SR=16000

# Target num channels
TARGET_NC=1

# Convert each file with SoX, and write the converted file
# to the corresponding output dir, preserving the internal
# structure of the input dir
find "$InDir" -type f -print0 | while read -d $'\0' input
do
  # echo "processing" $input

  # the output path, without the InDir prefix
  output=${input#$InDir}
  # # replace the original extension with .wav
  output=$OutDir$output

  # get the output directory, and create it if necessary
  # outdir=$(dirname "${output}")
  # mkdir -p "$outdir"

  # finally, convert the file
  sox "$input" -r $TARGET_SR "$output"

  # echo "saved as $output"
done