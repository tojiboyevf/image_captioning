#!/bin/bash

if [ ! -f "data/glove.6B.zip" ]
then
    wget "http://nlp.stanford.edu/data/glove.6B.zip" -O "data/glove.6B.zip"
fi

if [ ! -d "data/glove.6B" ]
then
    unzip "data/glove.6B.zip" -d "data/glove.6B"
fi
