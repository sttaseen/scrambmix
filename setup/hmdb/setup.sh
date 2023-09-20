#!/bin/bash

cd ../..


# Download the dataset
mkdir -p data
cd data
mkdir -p hmdb51
cd hmdb51
kaggle datasets download -d qxin258/hmdb51-rawframes
unzip hmdb51-rawframes.zip -d ./
cd ../..
