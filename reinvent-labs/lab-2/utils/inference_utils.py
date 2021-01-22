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

from utils import path_utils, midi_utils, display_utils

# --- local samples------------------------------------------------------------------

def load_melody_samples(n_sample=10):    
    """Load the samples used for evaluation."""
    
    sample_source_path = './dataset/eval.npy'
    
    data = np.load(sample_source_path)
    data = np.asarray(data, dtype=np.float32) # {-1, 1}

    random_idx = np.random.choice(len(data), n_sample, replace=False)
    
    sample_x = data[random_idx]

    sample_z = tf.random.truncated_normal((n_sample, 2, 8, 512))
    
    print("Loaded {} melody samples".format(len(sample_x)))

    return sample_x, sample_z

# --- Training ------------------------------------------------------------------

def generate_pianoroll(generator, conditioned_track, noise_vector=None):
    if noise_vector == None:
        noise_vector = tf.random.truncated_normal((1, 2, 8, 512))
    return generator((conditioned_track, noise_vector), training=False)


def generate_midi(generator, saveto_dir, input_midi_file='./Experiments/data/happy_birthday_easy.mid'):
    conditioned_track = midi_utils.get_conditioned_track(midi=input_midi_file)
    generated_pianoroll = generate_pianoroll(generator, conditioned_track)
    
    destination_path = path_utils.new_temp_midi_path(saveto_dir=saveto_dir)
    midi_utils.save_pianoroll_as_midi(generated_pianoroll.numpy(), destination_path=destination_path)
    return destination_path
    
def show_generated_pianorolls(generator, eval_dir, input_midi_file='./Experiments/data/happy_birthday_easy.mid', n_pr = 4):    
    conditioned_track = midi_utils.get_conditioned_track(midi=input_midi_file)
    for i in range(n_pr):
        generated_pianoroll = generate_pianoroll(generator, conditioned_track)
        display_utils.show_pianoroll(generated_pianoroll)