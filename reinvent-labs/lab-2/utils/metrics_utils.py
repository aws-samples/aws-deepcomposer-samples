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

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


# --- Metrics ------------------------------------------------------------------
class EmptyBarRateMetricCreator:
    label = "Empty Bar Rate"
    description = """The ratio of empty bars to the total number of bars."""
    
    def compute(self, pianoroll):
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        return np.mean(
            np.any(pianoroll > 0.5, (2, 3)).astype(np.float32), (0, 1))

class UniquePitchCountMetricCreator:
    label = "Unique Pitch Count"
    description = """The number of unique pitches used per bar."""
        
    def compute(self, pianoroll):
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        pitch_hist = np.mean(np.sum(pianoroll, 2), (0, 1))
        return np.linalg.norm(np.ones(pitch_hist.shape)-pitch_hist, axis=0) #Sums across each timestep in bar

class UniquePitchClassCountMetricCreator:
    label = "Unique Pitch Class Count"
    description = """
        Average number of notes in each bar after projecting to chroma space.
        Chroma features or Pitch Class Profiles are a distribution of the signal’s energy
        across a predefined set of pitch classes.
    """
    
    def _to_chroma(self, pianoroll):
        """Return the chroma features (not normalized)."""
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        remainder = pianoroll.shape[3] % 12
        if remainder:
            pianoroll = np.pad(
                pianoroll, ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0)))
        reshaped = np.reshape(
            pianoroll, (-1, pianoroll.shape[1], pianoroll.shape[2], 12,
                        pianoroll.shape[3] // 12 + int(remainder > 0),
                        pianoroll.shape[4]))
        return np.sum(reshaped, 4)
        
    def compute(self, pianoroll):
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        
        chroma_pianoroll = self._to_chroma(pianoroll)
        pitch_hist = np.mean(np.sum(chroma_pianoroll, 2), (0, 1))
        return np.linalg.norm(np.ones(pitch_hist.shape)-pitch_hist, axis=0) #Sums across each timestep in bar
    

class PolyphonicRateMetricCreator:
    label = "Polyphonic Rate"
    description = """
        The ratio of the number of time steps where the number of pitches
        being played is larger than `threshold` to the total number of time steps.
    """
    def __init__(self, threshold=2):
        self.threshold = threshold
        
    def compute(self, pianoroll):
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        n_poly = np.count_nonzero((np.count_nonzero(pianoroll, 3) > self.threshold), 2)
        return np.mean((n_poly / pianoroll.shape[2]), (0, 1))
    
class InScaleRateMetricCreator:
    label = "In Scale Rate"
    description = """
        The ratio of the average number of notes in a bar, which are in C major key which
        is the most common key found in music to the total number of notes.
    """
    
    def _to_chroma(self, pianoroll):
        """Return the chroma features (not normalized)."""
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        remainder = pianoroll.shape[3] % 12
        if remainder:
            pianoroll = np.pad(
                pianoroll, ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0)))
        reshaped = np.reshape(
            pianoroll, (-1, pianoroll.shape[1], pianoroll.shape[2], 12,
                        pianoroll.shape[3] // 12 + int(remainder > 0),
                        pianoroll.shape[4]))
        return np.sum(reshaped, 4)
    
    def _scale_mask(self, key=3):
            """Return a scale mask for the given key. Default to C major scale."""
            a_scale_mask = np.array([[[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]]], bool)
            return np.expand_dims(np.roll(a_scale_mask, -key, 2), -1).astype(np.float32)
        
    def compute(self, pianoroll):
        if len(pianoroll.shape) != 5:
            raise ValueError("Input pianoroll must have 5 dimensions.")
        
        chroma_pianoroll = self._to_chroma(pianoroll)
        in_scale = np.sum(self._scale_mask() * np.sum(chroma_pianoroll, 2), (0, 1, 2))
        return in_scale / np.sum(chroma_pianoroll, (0, 1, 2, 3))

class MusicQualityMetricsManager:
    
    def __init__(self, metrics_creators):
        self.metrics_creators = metrics_creators
        self.initialize()
    
    def initialize(self):
        self.metrics = {}
        for metrics_creator in self.metrics_creators:
            self.metrics[metrics_creator] = {'reference': None, 'per_iteration': []}
    
    def _reshape_pianoroll(self, pianoroll):
        return pianoroll.reshape((-1,2,16,128,4))[:,:,:,24:108,:]
    
    def append_metrics_for_iteration(self, pianoroll, iteration):
        reshaped_pianoroll = self._reshape_pianoroll(pianoroll)
        for metric_creator in self.metrics.keys():
            metric = metric_creator.compute(reshaped_pianoroll)
            self.metrics[metric_creator]['per_iteration'].append((iteration, metric))
    
    def set_reference_metrics(self, pianoroll):
        reshaped_pianoroll = self._reshape_pianoroll(pianoroll)
        for metric_creator in self.metrics.keys():
            metric = metric_creator.compute(reshaped_pianoroll)
            self.metrics[metric_creator]['reference'] = metric

    def plot_metrics(self):
        num_instruments = 4
        plt.ion()
        sns.set()
        fig, axs = plt.subplots(len(self.metrics), num_instruments, sharex=True, figsize=(60, 30))
        fig.tight_layout()
        plt.xscale('log')
        
        for instrument_idx in range(num_instruments):
            for metric_idx, metric_creator in enumerate(self.metrics):
                axs[metric_idx][instrument_idx].tick_params(axis='both', which='major', labelsize=30)
                axs[metric_idx][instrument_idx].tick_params(axis='both', which='minor', labelsize=30)
                
                metric_data = self.metrics[metric_creator]
                
                # Plot reference line
                axs[metric_idx][instrument_idx].plot(
                    [x[0] for x in metric_data['per_iteration']],
                    np.ones(len(metric_data['per_iteration'])) * metric_data['reference'][instrument_idx],
                    'r',
                    linewidth=10,
                    alpha=0.7
                )
                
                # Plot per-iteration metrics
                axs[metric_idx][instrument_idx].scatter(
                    [x[0] for x in metric_data['per_iteration']],
                    [x[1][instrument_idx] for x in metric_data['per_iteration']],
                    linewidth=10
                )
                

        for instrument_idx in range(num_instruments):
            label = "Iterations (Instrument {})".format(instrument_idx)
            axs[2][instrument_idx].set_xlabel(xlabel=label,fontsize=40)
        
        for metric_idx, metric_creator in enumerate(self.metrics):
            label = metric_creator.label
            axs[metric_idx][0].set_ylabel(ylabel=label,fontsize=40)
        
        plt.show()

        
        
DEFAULT_METRICS_CREATORS = [
    EmptyBarRateMetricCreator(),
    UniquePitchCountMetricCreator(),
    InScaleRateMetricCreator()
]
metrics_manager = MusicQualityMetricsManager(DEFAULT_METRICS_CREATORS)