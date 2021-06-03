# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import numpy as np
import pypianoroll

def save_pianoroll_as_midi(pianoroll,
                  programs=[0, 0, 0, 0],
                  is_drums=[False, False, False, False],
                  tempo=100,           # in bpm
                  beat_resolution=4,  # number of time steps
                  destination_path=None
                  ):
    
    pianoroll = pianoroll > 0

    # Reshape batched pianoroll array to a single pianoroll array
    pianoroll_ = pianoroll.reshape((-1, pianoroll.shape[2], pianoroll.shape[3]))

    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
        
    multitrack.write(destination_path)
    print('Midi saved to ', destination_path)
    return destination_path


def get_conditioned_track(midi=None, phrase_length=32, beat_resolution=4):
    
    if not isinstance(midi, str):
        # ----------- Generation from preprocessed dataset ------------------
        sample_x = midi
        sample_c = np.expand_dims(sample_x[..., 0], -1)
    else:
        # --------------- Generation from midi file -------------------------
        midi_file = midi
        parsed = pypianoroll.Multitrack(beat_resolution=beat_resolution)
        parsed.parse_midi(midi_file)

        sample_c = parsed.tracks[0].pianoroll.astype(np.float32)
        
        # Remove initial steps that have no note-on events
        first_non_zero = np.nonzero(sample_c.sum(axis=1))[0][0]
        
        # Use the first 'phrase_length' steps as the primer
        sample_c = sample_c[first_non_zero: first_non_zero + phrase_length]

        # Binarize data (ignore velocity value)
        sample_c[sample_c > 0] = 1
        sample_c[sample_c <= 0] = -1

        sample_c = np.expand_dims(np.expand_dims(sample_c, 0), -1)  # 1, 32, 128, 1
        
    sample_c = tf.convert_to_tensor(sample_c, dtype=tf.float32)
    return sample_c
