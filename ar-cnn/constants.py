# The MIT-Zero License

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

class Constants():
    # Make it a multiple of the batch size for best (balanced) performance
    samples_per_ground_truth_data_item = 8
    training_validation_split = 0.9
    # Number of Bars
    bars = 8
    # Number of Beats Per Bar
    beats_per_bar = 4
    beat_resolution = 4
    # number of bars to be shifted
    bars_shifted_per_sample = 4
    # Total number of pitches in a Pianoroll
    number_of_pitches = 128
    # Total number of Tracks
    number_of_channels = 1
    output_file_path = "outputs/output_{}.mid"
    tempo = 100  # 100 bpm
