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

import os
import numpy as np
import itertools
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import moviepy.editor as mpy
import music21
from IPython.display import Video
from IPython import display
import seaborn as sns
    
# --- plot------------------------------------------------------------------


def plot_loss_logs(G_loss, D_loss, figsize=(15, 5), smoothing=0.001):
    """Utility for plotting losses with smoothing."""
    plt.ion()
    sns.set()
    plt.figure(figsize=figsize)
    plt.plot(D_loss, label='C_loss')
    plt.plot(G_loss, label='G_loss')
    plt.legend(loc='lower right', fontsize='medium')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Losses', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    
    
def playmidi(filename):
    mf = music21.midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    s = music21.midi.translate.midiFileToStream(mf)
    s.show('midi')
    
def show_pianoroll(xs, min_pitch=45, max_pitch=85,
                   programs = [0, 0, 0, 0], save_dir=None):
    """ Plot a MultiTrack PianoRoll

    :param x: Multi Instrument PianoRoll Tensor
    :param min_pitch: Min Pitch / Min Y across all instruments.
    :param max_pitch: Max Pitch / Max Y across all instruments.
    :param programs: Program Number of the Tracks.
    :param file_name: Optional. File Name to save the plot.
    :return:
    """

    # Convert fake_x to numpy and convert -1 to 0
    xs = xs > 0

    channel_last = lambda x: np.moveaxis(np.array(x), 2, 0)
    xs = [channel_last(x) for x in xs]

    assert len(xs[0].shape) == 3, 'Pianoroll shape must have 3 dims, Got %d' % len(xs[0].shape)
    n_tracks, time_step, _ = xs[0].shape

    plt.ion()
    fig = plt.figure(figsize=(15, 4))
    
    x = xs[0]
    
    for j in range(4):
        b = j+1
        ax = fig.add_subplot(1,4,b)
        nz = np.nonzero(x[b-1])

        if programs:
            ax.set_xlabel('Time('+pretty_midi.program_to_instrument_class(programs[j%4])+')')

        if (j+1)== 1:
            ax.set_ylabel('Pitch')
        else:
            ax.set_yticks([])

        ax.scatter(nz[0], nz[1], s=np.pi * 3, color='bgrcmk'[b-1])
        ax.set_ylim(45, 85)
        ax.set_xlim(0, time_step)
        fig.add_subplot(ax)
        

def plot_pianoroll(iteration, xs, fake_xs, min_pitch=45, max_pitch=85,
                   programs = [0, 0, 0, 0], save_dir=None):
    """ Plot a MultiTrack PianoRoll

    :param x: Multi Instrument PianoRoll Tensor
    :param min_pitch: Min Pitch / Min Y across all instruments.
    :param max_pitch: Max Pitch / Max Y across all instruments.
    :param programs: Program Number of the Tracks.
    :param file_name: Optional. File Name to save the plot.
    :return:
    """

    # Convert fake_x to numpy and convert -1 to 0
    xs = xs > 0
    fake_xs = fake_xs > 0

    channel_last = lambda x: np.moveaxis(np.array(x), 2, 0)
    xs = [channel_last(x) for x in xs]
    fake_xs = [channel_last(fake_x) for fake_x in fake_xs]

    assert len(xs[0].shape) == 3, 'Pianoroll shape must have 3 dims, Got %d' % len(xs[0].shape)
    n_tracks, time_step, _ = xs[0].shape

    plt.ion()
    fig = plt.figure(figsize=(15, 8))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

    for i in range(4):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, n_tracks,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

        x, fake_x = xs[i], fake_xs[i]

        for j, (a, b) in enumerate(itertools.product([1, 2], [1, 2, 3, 4])):

            ax = fig.add_subplot(inner_grid[j])

            if a == 1:
                nz = np.nonzero(x[b-1])
            else:
                nz = np.nonzero(fake_x[b-1])

            if programs:
                ax.set_xlabel('Time('+pretty_midi.program_to_instrument_class(programs[j%4])+')')

            if b == 1:
                ax.set_ylabel('Pitch')
            else:
                ax.set_yticks([])

            ax.scatter(nz[0], nz[1], s=np.pi * 3, color='bgrcmk'[b-1])
            ax.set_ylim(45, 85)
            ax.set_xlim(0, time_step)
            fig.add_subplot(ax)

    if isinstance(iteration, int):
        plt.suptitle('iteration: {}'.format(iteration), fontsize=20)
        filename = os.path.join(save_dir, 'sample_iteration_%05d.png' % iteration)
    else:
        plt.suptitle('Inference', fontsize=20)
        filename = os.path.join(save_dir, 'sample_inference.png')
    plt.savefig(filename)
    plt.close(fig)

def display_loss(iteration, d_losses, g_losses):
    sns.set()
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(15,5))
    line1, = plt.plot(range(iteration+1), d_losses, 'r')
    line2, = plt.plot(range(iteration+1), g_losses, 'k')
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.legend((line1, line2), ('C-loss', 'G-loss'))
    
    return display.display(fig)


def make_training_video(folder_dir):
    files = sorted([os.path.join(folder_dir, f) for f in os.listdir(folder_dir) if f.endswith('.png')])
    frames = [mpy.ImageClip(f).set_duration(1) for f in files]  
    clip = mpy.concatenate_videoclips(frames, method="compose")
    clip.write_videofile("movie.mp4",fps=15) 
    return Video("movie.mp4")
    