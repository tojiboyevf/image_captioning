#!/bin/bash

mkdir ./data/flickr8k/
wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" -O "data/flickr8k/Flickr8k_Dataset.zip"
wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" -O "data/flickr8k/Flickr8k_text.zip"

unzip "data/flickr8k/Flickr8k_Dataset.zip" -d data/flickr8k/
unzip "data/flickr8k/Flickr8k_text.zip" -d data/flickr8k/Flickr8k_text
rm -r "data/flickr8k/Flickr8k_text/__MACOSX"
rm -r "data/flickr8k/__MACOSX"

wget "http://nlp.stanford.edu/data/glove.6B.zip" -O "data/glove.6B.zip"
unzip "data/glove.6B.zip" -d "data/glove.6B"
