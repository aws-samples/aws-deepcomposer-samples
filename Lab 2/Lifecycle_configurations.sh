#!/bin/bash

set -e

# OVERVIEW
# This script installs all necessary software for running the DeepComposer GAN notebook.

sudo -u ec2-user -i <<'EOF'
ENVIRONMENT=python3
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

conda update --all --y 
pip install tensorflow==1.14.0
pip install numpy==1.16.4
pip install pretty_midi
pip install pypianoroll
pip install music21
pip install seaborn
pip install --ignore-installed moviepy

source /home/ec2-user/anaconda3/bin/deactivate
EOF