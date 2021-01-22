# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import note_seq
import itertools
import functools
import numpy as np
import os

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
PERFORMANCE_VOCAB_PATH = os.path.join(_CURR_DIR, 'performance_vocab.txt')

MIN_PITCH, MAX_PITCH = 21, 108

class BaseVocab:
    """
    This class provides an abstraction over our vocabulary 
    Includes functions to access special tokens, length of vocabulary
    and retrieve tokens
    """
    def __init__(self, all_tokens):
        self._all_tokens = all_tokens
        self._map = dict()
        for i, token in enumerate(all_tokens):
            self._map[token] = i
        assert self._all_tokens[0] == "<S>"
        assert self._all_tokens[1] == "<PAD>"

    def idx_to_token(self, idx):
        return self._all_tokens[idx]

    @property
    def bos_token(self):
        return self._all_tokens[0]

    @property
    def pad_token(self):
        return self._all_tokens[1]

    @property
    def bos_id(self):
        return 0

    @property
    def pad_id(self):
        return 1

    @property
    def all_tokens(self):
        return self._all_tokens

    def token_to_idx(self, token):
        return self._map[token]

    def __len__(self):
        return len(self._all_tokens)

    def __getitem__(self, token):
        return self._map[token]

class DataAugmentationError(Exception):
    """
    Exception to be raised by augmentation functions on known failure.
    """
    pass

class ChordSymbolError(Exception):
    pass


def strip_ids(ids, ids_to_strip):
    """
    Strip ids_to_strip from the end ids.
    """
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def augment_note_sequence(ns, stretch_factor, transpose_amount, min_pitch, max_pitch):
    """
    Augment a NoteSequence by time stretch and pitch transposition.
    """
    augmented_ns = note_seq.sequences_lib.stretch_note_sequence(
        ns, stretch_factor, in_place=False)
    try:
        _, num_deleted_notes = note_seq.sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=min_pitch, max_allowed_pitch=max_pitch,
            in_place=True)
    except ChordSymbolError:
        print('Transposition of chord symbol(s) failed.')
    if num_deleted_notes:
        print('Transposition caused out-of-range pitch(es).')
    return augmented_ns


class PerformanceEventRepo(object):
    """
    Provides functionality to convert to and from a MIDI to a Performance notesequence used in
    https://arxiv.org/abs/1808.03715
    
    Also provides additional functionality to augment the notesequence, filter pitches, and convert 
    to and from a text format 
    """
    def __init__(self, steps_per_second=100, num_velocity_bins=32, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH,
                 stretch_factors=[1.0], pitch_transpose_lower=0, pitch_transpose_upper=0):

        self._steps_per_second = steps_per_second
        self._num_velocity_bins = num_velocity_bins
        
        # Load the Performance vocab
        f = open(PERFORMANCE_VOCAB_PATH, "r")
        self.contents = f.readlines()
        
        self.ids_to_events = {key: value.strip() for key, value in enumerate(self.contents)}
        self.events_to_ids = {value.strip(): key for key, value in enumerate(self.contents)}

        self.stretch_factors = stretch_factors
        self.transpose_amounts = list(range(pitch_transpose_lower,
                                            pitch_transpose_upper + 1))

        self.augment_params = itertools.product(
            self.stretch_factors, self.transpose_amounts)
        self.augment_fns = [
            functools.partial(augment_note_sequence,
                              stretch_factor=s, transpose_amount=t, min_pitch=min_pitch, max_pitch=max_pitch)
            for s, t in self.augment_params
        ]
        self.min_pitch, self.max_pitch = min_pitch, max_pitch

    def filter_pitches(self, ns):
        """
        Filter notes in note sequence to keep notes that lie between MIN_PITCH and MAX_PITCH
        """
        new_note_list = []
        deleted_note_count = 0
        end_time = 0

        for note in ns.notes:
            if self.min_pitch <= note.pitch <= self.max_pitch:
                end_time = max(end_time, note.end_time)
                new_note_list.append(note)
            else:
                deleted_note_count += 1

        if deleted_note_count > 0:
            del ns.notes[:]
            ns.notes.extend(new_note_list)

        ns.total_time = end_time

    def encode_event(self, event):

        event_name = None
        if event.event_type == note_seq.performance_lib.PerformanceEvent.NOTE_ON:
            event_name = f"NOTE_ON_{event.event_value}" 
        elif event.event_type == note_seq.performance_lib.PerformanceEvent.NOTE_OFF:
            event_name = f"NOTE_OFF_{event.event_value}" 
        elif event.event_type == note_seq.performance_lib.PerformanceEvent.TIME_SHIFT:
            event_name = f"TIME_SHIFT_{event.event_value}" 
        elif event.event_type == note_seq.performance_lib.PerformanceEvent.VELOCITY:
            event_name = f"VELOCITY_{event.event_value}"  
            
        if event_name:
            return self.events_to_ids[event_name]
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")

    def decode_event(self, index):
        try:
            event_name = self.ids_to_events[index]
            event_splits = event_name.split('_')
            event_type, event_value = '_'.join(event_splits[:-1]), int(event_splits[-1])
            if event_type == 'NOTE_ON':
                return note_seq.performance_lib.PerformanceEvent(
                    event_type=note_seq.performance_lib.PerformanceEvent.NOTE_ON, event_value=event_value)
            elif event_type == 'NOTE_OFF':
                return note_seq.performance_lib.PerformanceEvent(
                    event_type=note_seq.performance_lib.PerformanceEvent.NOTE_OFF, event_value=event_value)
            elif event_type == 'TIME_SHIFT':
                return note_seq.performance_lib.PerformanceEvent(
                    event_type=note_seq.performance_lib.PerformanceEvent.TIME_SHIFT, event_value=event_value)
            elif event_type == 'VELOCITY':
                return note_seq.performance_lib.PerformanceEvent(
                    event_type=note_seq.performance_lib.PerformanceEvent.VELOCITY, event_value=event_value)
        except:
            raise ValueError('Unknown event index: %s' % index)

    def encode_note_sequence(self, ns):
        """
        Transform a NoteSequence into a list of performance event indices.
        Args:
          ns: NoteSequence proto containing the performance to encode.
        Returns:
          ids: List of performance event indices.
        """
        performance = note_seq.performance_lib.Performance(
            note_seq.quantize_note_sequence_absolute(
                ns, self._steps_per_second),
            num_velocity_bins=self._num_velocity_bins)

        event_ids = [self.encode_event(event) for event in performance]

        return event_ids

    def encode_transposition(self, input_midi):
        """
        Augment and transform a MIDI file into a list of performance event indices.
        Args:
          input_midi: Path to input MIDI file
        Returns:
          ids: List of performance event indices.
        """
        if input_midi:
            ns = note_seq.midi_file_to_sequence_proto(input_midi)
            ns = note_seq.sequences_lib.apply_sustain_control_changes(ns)
            del ns.control_changes[:]
        else:
            ns = note_seq.protobuf.music_pb2.NoteSequence()

        for augment_fn in self.augment_fns:
            # Augment and encode the performance.
            try:
                # print(augment_fn)
                augmented_performance_sequence = augment_fn(ns)
            except DataAugmentationError:
                # print(DataAugmentationError)
                continue
            yield self.encode_note_sequence(augmented_performance_sequence)

    def encode(self, input_midi):
        """
        Transform a MIDI filename into a list of performance event indices.
        Args:
          input_midi: Path to the MIDI file.
        Returns:
          ids: List of performance event indices.
        """
        if input_midi:
            ns = note_seq.midi_file_to_sequence_proto(input_midi)
            ns = note_seq.sequences_lib.apply_sustain_control_changes(ns)
            del ns.control_changes[:]
        else:
            ns = note_seq.protobuf.music_pb2.NoteSequence()
        self.filter_pitches(ns)

        return self.encode_note_sequence(ns)

    def decode(self, event_ids, save_path=None):
        """
        Transform a sequence of event indices into a performance MIDI file.
        Args:
          event_ids: List of performance event indices.
        Returns:
          Path to the temporary file where the MIDI was saved.
        """
        performance = note_seq.performance_lib.Performance(
            quantized_sequence=None,
            steps_per_second=self._steps_per_second,
            num_velocity_bins=self._num_velocity_bins)

        tokens = []
        for i, event_id in enumerate(event_ids):
            if len(tokens) >= 2 and self.ids_to_events[tokens[-1]] == 'TIME_SHIFT_100' and self.ids_to_events[
                tokens[-1]] == 'TIME_SHIFT_100' and \
                    self.ids_to_events[event_id] == 'TIME_SHIFT_100':
                continue
            tokens.append(event_id)
            
            if event_id>1:
                performance.append(self.decode_event(event_id))
            
        ns = performance.to_sequence(max_note_duration=3)
        note_seq.sequence_proto_to_midi_file(ns, save_path)

        return save_path

    def create_vocab_txt(self, input_dir):
        event2word = [value[:-1] for value in self.contents]
        with open(os.path.join(input_dir, "vocab.txt"), 'w') as f:
            f.write("\n".join(event2word))

    def midi_quantizer(self, input_midi, output_midi):
        """
        Transform a MIDI filename into a list of performance event indices.
        Args:
          s: Path to the MIDI file.
        Returns:
          ids: List of performance event indices.
        """
        if input_midi:
            ns = note_seq.midi_file_to_sequence_proto(input_midi)
            ns = note_seq.sequences_lib.apply_sustain_control_changes(ns)
            del ns.control_changes[:]
        else:
            ns = note_seq.protobuf.music_pb2.NoteSequence()
        note_seq.sequence_proto_to_midi_file(ns, output_midi)
        return output_midi

    def to_text(self, input_midi, output_txt):
        ids = self.encode(input_midi)
        event_text = [self.ids_to_events[idx] for idx in ids]
        with open(output_txt, 'w') as f:
            f.write("\n".join(event_text))

    def to_text_transposition(self, input_midi, output_txt):
        for i, ids in enumerate(self.encode_transposition(input_midi)):
            event_text = [self.ids_to_events(idx) for idx in ids]
            filename, ext = os.path.splitext(output_txt)
            with open(filename + '_arg' + str(i) + '.txt', 'w') as f:
                f.write("\n".join(event_text))

    def from_text(self, input_txt, output_midi):
        with open(input_txt, 'r', encoding='utf-8') as f:
            events = f.read().strip().splitlines()
        ids = [self.events_to_ids[event] for event in events]
        return self.decode(ids, save_path=output_midi)

    def to_npy_transposition(self, input_midi, out_npy_file):
        for i, event_ids in enumerate(self.encode_transposition(input_midi)):
            filename, ext = os.path.splitext(out_npy_file)
            event_ids_np = np.array(event_ids, dtype=np.int32)
            np.save(filename + '_arg' + str(i) + '.npy', event_ids_np)

    def to_npy(self, input_midi, out_npy_file):
        event_ids = self.encode(input_midi)
        np.save(out_npy_file, np.array(event_ids, dtype=np.int32))

    def npy_to_midi(self, in_npy_file, out_midi_file):
        event_ids = np.load(in_npy_file)
        return self.decode(event_ids, save_path=out_midi_file)