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

import traceback
import numpy as np
import keras
from augmentation import AddAndRemoveAPercentageOfNotes


class PianoRollGenerator(keras.utils.Sequence):
    def __init__(self, sample_list, batch_size, bars, samples_per_data_item,
                 beat_resolution, number_of_pitches, number_of_channels,
                 beats_per_bar, sampling_lower_bound_remove,
                 sampling_upper_bound_remove, sampling_lower_bound_add,
                 sampling_upper_bound_add):

        self.sample_list = sample_list
        self.batch_size = batch_size
        self.bars = bars
        self.number_of_pitches = number_of_pitches
        self.number_of_channels = number_of_channels
        self.samples_per_data_item = samples_per_data_item
        self.sample_index = 0
        self.beat_resolution = beat_resolution
        self.beats_per_bar = beats_per_bar
        self.sampling_lower_bound_remove = sampling_lower_bound_remove
        self.sampling_upper_bound_remove = sampling_upper_bound_remove
        self.sampling_lower_bound_add = sampling_lower_bound_add
        self.sampling_upper_bound_add = sampling_upper_bound_add

    def generate_training_pairs(self):
        '''
        Generates Training Pairs till @training_input / @training_target have @batch_size files.
        '''

        # Create the training and target lists
        training_input = []
        training_target = []
        while len(training_input) <= self.batch_size:
            target_pianoroll = self.sample_list[self.sample_index]
            self.sample_index = (self.sample_index + 1) % len(self.sample_list)
            try:
                training_data_shape = (self.bars * self.beats_per_bar *
                                       self.beat_resolution,
                                       self.number_of_pitches,
                                       self.number_of_channels)

                # For each pianoroll section, add or remove certain percentage of notes
                add_remove_notes = AddAndRemoveAPercentageOfNotes(
                    sampling_lower_bound_remove=self.
                    sampling_lower_bound_remove,
                    sampling_upper_bound_remove=self.
                    sampling_upper_bound_remove,
                    sampling_lower_bound_add=self.sampling_lower_bound_add,
                    sampling_upper_bound_add=self.sampling_upper_bound_add)

                input_pianorolls = add_remove_notes.sample(
                    target_pianoroll, self.samples_per_data_item)

                for input_pianoroll in input_pianorolls:
                    training_input.append(
                        input_pianoroll.reshape(training_data_shape))
                    xor_target = np.logical_xor(input_pianoroll,
                                                target_pianoroll)
                    training_target.append(
                        xor_target.reshape(training_data_shape))

                if len(training_input) >= self.batch_size:
                    training_input = np.asarray(
                        training_input[:self.batch_size])
                    training_target = np.asarray(
                        training_target[:self.batch_size])
                    return training_input, training_target

            except Exception as e:
                print('Error generating input and target pair')
                traceback.print_exc()

    def __getitem__(self, index):
        '''Generates 1 batch of data'''
        training_input, training_target = self.generate_training_pairs()
        return training_input, training_target

    def __len__(self):
        '''Number of batches / epoch'''
        samples_to_generate = int(
            (len(self.sample_list) * self.samples_per_data_item) /
            self.batch_size)
        return samples_to_generate
