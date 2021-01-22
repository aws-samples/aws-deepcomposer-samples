#!/bin/sh
source activate python3
conda update --all --y 
pip install tensorflow-gpu==1.14.0
pip install numpy==1.16.4
pip install pretty_midi
pip install pypianoroll
pip install music21
pip install seaborn
pip install --ignore-installed moviepy
