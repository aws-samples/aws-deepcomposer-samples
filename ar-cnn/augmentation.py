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

import itertools
import copy
import random
import math
import numpy as np


class AddAndRemoveAPercentageOfNotes():
    def __init__(self, sampling_lower_bound_remove,
                 sampling_upper_bound_remove, sampling_lower_bound_add,
                 sampling_upper_bound_add):

        self.sampling_lower_bound_remove = sampling_lower_bound_remove
        self.sampling_upper_bound_remove = sampling_upper_bound_remove
        self.sampling_lower_bound_add = sampling_lower_bound_add
        self.sampling_upper_bound_add = sampling_upper_bound_add

    def apply_augmentation_to_sample(self, piano_roll):
        '''
        Randomly adds and removes percentages of notes.
        '''
        sampling_percentage_remove = np.random.random_integers(
            self.sampling_lower_bound_remove, self.sampling_upper_bound_remove)
        sampling_percentage_add = np.random.uniform(
            self.sampling_lower_bound_add, self.sampling_upper_bound_add)
        # Removing certain values of nonzero indices
        # based on the presence and absence of number of notes
        rows_remove_notes, columns_remove_notes = self.create_notes_mask(
            piano_roll, sampling_percentage_remove, notes_exists=True)
        # Adding certain values from zeros index based on the
        # the presence and absence of number of notes
        rows_add_notes, columns_add_notes = self.create_notes_mask(
            piano_roll, sampling_percentage_add, notes_exists=False)
        result = copy.deepcopy(piano_roll)
        result[rows_remove_notes, columns_remove_notes] = 0
        result[rows_add_notes, columns_add_notes] = 1
        return result

    def create_notes_mask(self,
                          piano_roll,
                          sampling_percentage,
                          notes_exists=True):
        if notes_exists:
            # Get the indices from pianoroll where note exists
            indices = np.nonzero(piano_roll)
        else:
            # Get the indices from pianoroll where note doesn't exist
            indices = np.nonzero(piano_roll == 0)
        # Create num_of_notes - an array of True values
        # of length = number of non-zero/zero values
        # in pianoroll based on presensce or absence of notes
        num_notes = np.full(len(indices[0]), True)
        # If any of the sampling percentage is 10,
        # then we choose 90% of the true values to become false
        num_notes[:math.
                  floor(len(indices[0]) *
                        (1 - sampling_percentage / 100))] = False
        random.shuffle(num_notes)
        rows_modified_notes = list(itertools.compress(indices[0], num_notes))
        columns_modified_notes = list(itertools.compress(
            indices[1], num_notes))
        return rows_modified_notes, columns_modified_notes

    def sample(self, piano_roll, number_of_samples):
        result = []
        for _ in range(number_of_samples):
            result.append(self.apply_augmentation_to_sample(piano_roll))
        return result
