#!/bin/bash

cd ../..


# Download the dataset
mkdir -p data
cd data
mkdir -p hmdb51
cd hmdb51
kaggle datasets download -d qxin258/HMDB51
unzip HMDB51.zip -d ./
cd ../..
