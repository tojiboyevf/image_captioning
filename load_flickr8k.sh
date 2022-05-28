#!/bin/bash
CURRENT_FILE_DIR="$( dirname "${BASH_SOURCE[0]}" )"

mkdir -p "$CURRENT_FILE_DIR/data/flickr8k"

if [ ! -f "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_Dataset.zip" ]
then
    wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" -O "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_Dataset.zip"
    wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" -O "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_text.zip"
fi

if [ ! -d "$CURRENT_FILE_DIR/data/flickr8k/Flicker8k_Dataset" ]
then
    unzip "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_Dataset.zip" -d "$CURRENT_FILE_DIR/data/flickr8k/"
fi

if [ ! -d "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_text" ]
then
    unzip "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_text.zip" -d "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_text"
    rm -r "$CURRENT_FILE_DIR/data/flickr8k/Flickr8k_text/__MACOSX"
fi

if [ -d "$CURRENT_FILE_DIR/data/flickr8k/__MACOSX" ]
then
    rm -r "$CURRENT_FILE_DIR/data/flickr8k/__MACOSX"
fi
