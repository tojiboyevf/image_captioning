#!/bin/bash

mkdir -p data/flickr8k/
if [ ! -f "data/flickr8k/Flickr8k_Dataset.zip" ]
then
    wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" -O "data/flickr8k/Flickr8k_Dataset.zip"
    wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" -O "data/flickr8k/Flickr8k_text.zip"
fi

if [ ! -d "data/flickr8k/Flicker8k_Dataset" ]
then
    unzip "data/flickr8k/Flickr8k_Dataset.zip" -d data/flickr8k/
fi

if [ ! -d "data/flickr8k/Flickr8k_text" ]
then
    unzip "data/flickr8k/Flickr8k_text.zip" -d data/flickr8k/Flickr8k_text
    rm -r "data/flickr8k/Flickr8k_text/__MACOSX"
fi

if [ -d "data/flickr8k/__MACOSX" ]
then
    rm -r "data/flickr8k/__MACOSX"
fi
