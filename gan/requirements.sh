#!/bin/sh
source activate python3
conda update --all --y 
pip install tensorflow-gpu==1.14.0
pip install numpy==1.16.4
pip install pretty_midi==0.2.9
pip install pypianoroll==0.5.3
pip install music21==6.7.1
pip install seaborn==0.11.1
pip install --ignore-installed moviepy==1.0.3
