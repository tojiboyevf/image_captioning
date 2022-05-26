#!/bin/bash

mkdir -p data/coco saved_models

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

unzip annotations_trainval2014.zip -d data/coco
unzip val2014.zip -d data/coco

rm val2014.zip annotations_trainval2014.zip