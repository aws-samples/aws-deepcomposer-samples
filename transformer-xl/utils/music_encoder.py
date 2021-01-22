# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append("./utils")

from performance_event_repo import PerformanceEventRepo
from midi_utils import find_files_by_extensions
import functools
import time
import os
import pandas as pd
import logging
import multiprocessing as mpl

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


def get_midi_paths(dataset_dir):
    if not os.path.exists(dataset_dir):
        raise ValueError('Cannot find dataset. Please provide the correct path')
        
    train_paths = [os.path.join(dataset_dir,'train', midi) for midi in os.listdir(os.path.join(dataset_dir,'train'))]
    validation_paths = [os.path.join(dataset_dir,'valid', midi) for midi in os.listdir(os.path.join(dataset_dir,'valid'))]
    test_paths = [os.path.join(dataset_dir,'test', midi) for midi in os.listdir(os.path.join(dataset_dir,'test'))]

    return train_paths, validation_paths, test_paths


class MusicEncoder:
    def __init__(self):
        pass

    def build_encoder(self, algorithm, stretch_factors=[0.95,0.975,1.0,1.025,1.05], 
                      pitch_transpose = (-3,3)):
        if algorithm == 'performance':
            pitch_transpose_lower, pitch_transpose_upper = pitch_transpose
            self.encoder = PerformanceEventRepo(steps_per_second=100, num_velocity_bins=32,
                                                stretch_factors=stretch_factors,
                                                pitch_transpose_lower=pitch_transpose_lower,
                                                pitch_transpose_upper=pitch_transpose_upper)
        
        else:
            print("This algorithm is not currently supported")
            raise NotImplementedError
        

    def run_to_text(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.to_text(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_text_with_transposition(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.to_text_transposition(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_npy(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.to_npy(path, os.path.join(out_dir, filename + '.npy'))


    def run_to_npy_with_transposition(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.to_npy_transposition(path, os.path.join(out_dir, filename + '.npy'))


    def run_from_text(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.from_text(path, os.path.join(out_dir, filename + '.mid'))


    def run_npy_to_midi(self, path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        self.encoder.npy_to_midi(path, os.path.join(out_dir, filename + '.mid'))

    def convert(self, input_folder, output_folder, mode):
        num_cpus = mpl.cpu_count()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if mode == 'to_txt' or mode == 'midi_to_npy':
            if mode == 'to_txt':
                converted_format = 'txt'
                convert_transposition_f = self.run_to_text_with_transposition
                convert_f = self.run_to_text
            else:
                converted_format = 'npy'
                convert_transposition_f = self.run_to_npy_with_transposition
                convert_f = self.run_to_npy

            print('Converting midi files from {} to {}...'
                  .format(input_folder, converted_format))



            train_paths, valid_paths, test_paths = get_midi_paths(input_folder)
            print('Loaded dataset from {}. Train/Val/Test={}/{}/{}'
                      .format(input_folder, len(train_paths), len(valid_paths),
                              len(test_paths)))

            for split_name, midi_paths in [('train', train_paths),
                                               ('valid', valid_paths),
                                               ('test', test_paths)]:
                if split_name == 'train':
                    convert_function = convert_transposition_f
                else:
                    convert_function = convert_f

                out_split_dir = os.path.join(output_folder, split_name)
                os.makedirs(out_split_dir, exist_ok=True)
                start = time.time()

                with mpl.Pool(num_cpus - 1) as pool:
                    pool.map(functools.partial(convert_function, out_dir=out_split_dir),
                                 midi_paths)
                print('Split {} converted! Spent {}s to convert {} samples.'
                          .format(split_name, time.time() - start, len(midi_paths)))

            self.encoder.create_vocab_txt(output_folder)

        elif mode == 'to_midi' or mode == 'npy_to_midi':
            convert_f = self.run_from_text if mode == 'to_midi' else self.run_npy_to_midi
            start = time.time()
            if mode == 'npy_to_midi':
                input_paths = list(find_files_by_extensions(input_folder, ['.npy']))
            else:
                input_paths = list(find_files_by_extensions(input_folder, ['.txt']))
                
            with mpl.Pool(num_cpus - 1) as pool:
                pool.map(functools.partial(convert_f,
                                           out_dir=output_folder),
                         input_paths)
            print('Test converted! Spent {}s to convert {} samples.'
                  .format(time.time() - start, len(input_paths)))
        else:
            raise NotImplementedError