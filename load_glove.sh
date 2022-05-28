#!/bin/bash
CURRENT_FILE_DIR="$( dirname "${BASH_SOURCE[0]}" )"

if [ ! -f "$CURRENT_FILE_DIR/data/glove.6B.zip" ]
then
    wget "http://nlp.stanford.edu/data/glove.6B.zip" -O "$CURRENT_FILE_DIR/data/glove.6B.zip"
fi

if [ ! -d "$CURRENT_FILE_DIR/data/glove.6B" ]
then
    unzip "$CURRENT_FILE_DIR/data/glove.6B.zip" -d "$CURRENT_FILE_DIR/data/glove.6B"
fi
