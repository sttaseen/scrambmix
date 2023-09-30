#!/bin/bash

cd ../..

# Set parameters
n_workers=32

# Download the dataset
mkdir -p data
cd data
mkdir -p something-something-v2
cd something-something-v2
kaggle datasets download -d sttaseen/something-something-v2
unzip something-something-v2.zip -d ./

cd ../..
# Split the videos
python tools/something-something-v2/split_videos.py -d data/something-something-v2/something-something-v2
# Extra raw frames
python mmaction2/tools/data/build_rawframes.py data/something-something-v2/something-something-v2/test data/something-something-v2/rawframes/test --ext webm --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/something-something-v2/something-something-v2/train data/something-something-v2/rawframes/train --ext webm --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv
python mmaction2/tools/data/build_rawframes.py data/something-something-v2/something-something-v2/val data/something-something-v2/rawframes/val --ext webm --task rgb --level 1 --num-worker $n_workers --out-format jpg --use-opencv

# # Build the labels
python tools/something-something-v2/build_labels.py data/something-something-v2


