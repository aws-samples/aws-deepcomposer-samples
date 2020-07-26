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
import logging
import pypianoroll
import keras
import numpy as np
from losses import Loss
from constants import Constants
import copy

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, model=None):
        self.model = model
        self.number_of_timesteps = (Constants.beat_resolution *
                                    Constants.beats_per_bar * Constants.bars)

    def load_model(self, model_path):
        """
        Loads a trained keras model

        Parameters
        ----------
        model_path : string
            Full file path to the trained model

        Returns
        -------
        None
        """
        self.model = keras.models.load_model(model_path,
                                             custom_objects={
                                                 'built_in_softmax_kl_loss':
                                                 Loss.built_in_softmax_kl_loss
                                             },
                                             compile=False)

    @staticmethod
    def convert_tensor_to_midi(tensor, tempo, output_file_path):
        """
        Writes a pianoroll tensor to a midi file

        Parameters
        ----------
        tensor : 2d numpy array
            pianoroll to be converted to a midi
        tempo : float
            tempo to output
        output_file_path : str
            output midi file path

        Returns
        -------
        None
        """

        single_track = pypianoroll.Track(pianoroll=tensor)
        multi_track = pypianoroll.Multitrack(
            tracks=[single_track],
            tempo=tempo,
            beat_resolution=Constants.beat_resolution)
        output_file_index = 0
        while os.path.isfile(output_file_path.format(output_file_index)):
            output_file_index += 1
        multi_track.write(output_file_path.format(output_file_index))

    @staticmethod
    def get_indices(input_tensor, value):
        """
        Parameters
        ----------
        input_tensor : 2d numpy array
        value : int (either 1 or 0)

        Returns
        -------
        indices_with_value : 2d array of indices in the input_tensor where the pixel value equals value (1 or 0).
        """
        indices_with_value = np.argwhere(input_tensor.astype(np.bool_) == value)
        return set(map(tuple, indices_with_value))

    @staticmethod
    def get_softmax(input_tensor, temperature):
        """
        Gets the softmax of a tensor with temperature

        Parameters
        ----------
        input_tensor : numpy array
            original tensor (e.g. original predictions)
        temperature : int
            softmax temperature

        Returns
        -------
        tensor : numpy array
            softmax of input tensor with temperature
        """
        tensor = input_tensor / temperature
        tensor = np.exp(tensor)
        tensor = tensor / np.sum(tensor)
        return tensor

    @staticmethod
    def get_sampled_index(input_tensor):
        """
        Gets a randomly chosen index from the input tensor

        Parameters
        ----------
        input_tensor : numpy array
            original tensor
        Returns
        -------
        tensor : numpy array
            softmax of input tensor with temperature
        """

        sampled_index = np.random.choice(range(input_tensor.size),
                                         1,
                                         p=input_tensor.ravel())
        sampled_index = np.unravel_index(sampled_index, input_tensor.shape)
        return sampled_index

    def generate_composition(self, input_midi_path, inference_params):
        """
        Generates a new composition based on an old midi

        Parameters
        ----------
        input_midi_path : str
            input midi path
        inference_params : json
            JSON with inference parameters

        Returns
        -------
        None
        """
        try:
            input_tensor = self.convert_midi_to_tensor(input_midi_path)
            output_tensor = self.sample_multiple(
                input_tensor, inference_params['temperature'],
                inference_params['maxPercentageOfInitialNotesRemoved'],
                inference_params['maxNotesAdded'],
                inference_params['samplingIterations'])
            self.convert_tensor_to_midi(output_tensor, Constants.tempo,
                                        Constants.output_file_path)
        except Exception:
            logger.error("Unable to generate composition.")
            raise

    def convert_midi_to_tensor(self, input_midi_path):
        """
        Converts a midi to pianoroll tensor

        Parameters
        ----------
        input_midi_path : string
            Full file path to the input midi

        Returns
        -------
        2d numpy array
            2d tensor that is a pianoroll
        """

        multi_track = pypianoroll.Multitrack(
            beat_resolution=Constants.beat_resolution)
        try:
            multi_track.parse_midi(input_midi_path,
                                   algorithm='custom',
                                   first_beat_time=0)
        except Exception as e:
            logger.error("Failed to parse the MIDI file.")

        if len(multi_track.tracks) > 1:
            logger.error("Input MIDI file has more than 1 track.")

        multi_track.pad_to_multiple(self.number_of_timesteps)
        multi_track.binarize()
        pianoroll = multi_track.tracks[0].pianoroll

        if pianoroll.shape[0] > self.number_of_timesteps:
            logger.error("Input MIDI file is longer than 8 bars.")

        # truncate
        tensor = pianoroll[0:self.number_of_timesteps, ]
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.expand_dims(tensor, axis=3)

        return tensor

    def mask_not_allowed_notes(self, current_input_indices, output_tensor):
        """
        Masks notes in output tensor that cannot be added or removed

        Parameters
        ----------
        current_input_indices : 2d numpy array
          indices to be masked based on the current input that was fed to model
        output_tensor : 2d numpy array
          consists of probabilities that are predicted by the model

        Returns
        -------
        2d numpy array - output tensor with not allowed notes masked
        """

        if len(current_input_indices) != 0:
            output_tensor[tuple(np.asarray(list(current_input_indices)).T)] = 0
            if np.count_nonzero(output_tensor) != 0:
                output_tensor = output_tensor / np.sum(output_tensor)
        return output_tensor

    def sample_multiple(self, input_tensor, temperature,
                        max_removal_percentage, max_notes_to_add,
                        number_of_iterations):
        """
        Samples multiple times from an tensor.
        Returns the final output tensor after X number of iterations.

        Parameters
        ----------
        input_tensor : 2d numpy array
            original tensor (i.e. user input melody)
        temperature : float
            temperature to apply before softmax during inference
        max_removal_percentage : float
            maximum percentage of notes that can be removed from the original input
        max_notes_to_add : int
            maximum number of notes that can be added to the original input
        number_of_iterations : int
            number of iterations to sample from the model predictions

        Returns
        -------
        2d numpy array
            output tensor (i.e. new composition)
        """

        max_original_notes_to_remove = int(
            max_removal_percentage * np.count_nonzero(input_tensor) / 100)
        notes_removed_count = 0
        notes_added_count = 0

        original_input_one_indices = self.get_indices(input_tensor, 1)
        original_input_zero_indices = self.get_indices(input_tensor, 0)

        current_input_one_indices = copy.deepcopy(original_input_one_indices)
        current_input_zero_indices = copy.deepcopy(original_input_zero_indices)

        for _ in range(number_of_iterations):
            input_tensor, notes_removed_count, notes_added_count = self.sample_notes_from_model(
                input_tensor, max_original_notes_to_remove, max_notes_to_add,
                temperature, notes_removed_count, notes_added_count,
                original_input_one_indices, original_input_zero_indices,
                current_input_zero_indices, current_input_one_indices)

        return input_tensor.reshape(self.number_of_timesteps,
                                    Constants.number_of_pitches)

    def sample_notes_from_model(self,
                                input_tensor,
                                max_original_notes_to_remove,
                                max_notes_to_add,
                                temperature,
                                notes_removed_count,
                                notes_added_count,
                                original_input_one_indices,
                                original_input_zero_indices,
                                current_input_zero_indices,
                                current_input_one_indices,
                                num_notes=1):
        """
        Generates a sample from the tensor and return a new tensor
        Modifies current_input_zero_indices, current_input_one_indices, and input_tensor

        Parameters
        ----------
        input_tensor : 2d numpy array
            input tensor to feed into the model
        max_original_notes_to_remove : int
            maximum number of notes to remove from the original input
        max_notes_to_add : int
            maximum number of notes that can be added to the original input
        temperature : float
            temperature to apply before softmax during inference
        notes_removed_count : int
            number of original notes that have been removed from input
        notes_added_count : int
            number of new notes that have been added to the input
        original_input_one_indices : set of tuples
            indices which have value 1 in original input
        original_input_zero_indices : set of tuples
            indices which have value 0 in original input
        current_input_zero_indices : set of tuples
            indices which have value 0 and were not part of the original input
        current_input_one_indices : set of tuples
            indices which have value 1 and were part of the original input

        Returns
        -------
        input_tensor : 2d numpy array
            output after samping from the model prediction
        notes_removed_count : int
            updated number of original notes removed
        notes_added_count : int
            updated number of new notes added
        """

        output_tensor = self.model.predict([input_tensor])

        # Apply temperature and softmax
        output_tensor = self.get_softmax(output_tensor, temperature)

        if notes_removed_count >= max_original_notes_to_remove:
            # Mask all pixels that both have a note and were once part of the original input
            output_tensor = self.mask_not_allowed_notes(current_input_one_indices, output_tensor)

        if notes_added_count > max_notes_to_add:
            # Mask all pixels that both do not have a note and were not once part of the original input
            output_tensor = self.mask_not_allowed_notes(current_input_zero_indices, output_tensor)

        if np.count_nonzero(output_tensor) == 0:
            return input_tensor, notes_removed_count, notes_added_count

        sampled_index = self.get_sampled_index(output_tensor)
        sampled_index_transpose = tuple(np.array(sampled_index).T[0])

        if input_tensor[sampled_index]:
            # Check if the note being removed is from the original input
            if notes_removed_count < max_original_notes_to_remove and (
                sampled_index_transpose in original_input_one_indices):
                notes_removed_count += 1
                current_input_one_indices.remove(sampled_index_transpose)
            elif tuple(sampled_index_transpose) not in original_input_one_indices:
                notes_added_count -= 1
                current_input_zero_indices.add(sampled_index_transpose)
            input_tensor[sampled_index] = 0
        else:
            # Check if the note being added is not in original input
            if sampled_index_transpose not in original_input_one_indices:
                notes_added_count += 1
                current_input_zero_indices.remove(sampled_index_transpose)
            else:
                notes_removed_count -= 1
                current_input_one_indices.add(sampled_index_transpose)
            input_tensor[sampled_index] = 1
        input_tensor = input_tensor.astype(np.bool_)
        return input_tensor, notes_removed_count, notes_added_count
