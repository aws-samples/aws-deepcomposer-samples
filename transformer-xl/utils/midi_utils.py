# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from music21 import midi
import pypianoroll
from pypianoroll import Multitrack
from texttable import Texttable
import os
from pprint import pprint

def play_midi(input_midi):
    '''Takes path to an input and plays the midi file in the notebook cell
    :param input_midi: Path to midi file
    :return:
    '''
    midi_object = midi.MidiFile()
    midi_object.open(input_midi)
    midi_object.read()
    midi_object.close()
    show_midi = midi.translate.midiFileToStream(midi_object)
    show_midi.show('midi')

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False

    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)
                
def print_sample_array(split, parent_dir="data/jsb_chorales_numpy"):
    """
    Prints a randomly sampled numpy array from the parent_dir
    """

    midi_files = [
        os.path.join(parent_dir, split, midi)
        for midi in os.listdir(os.path.join(parent_dir, split))
    ]
    selection = np.random.choice(midi_files)
    pprint(np.load(selection))
