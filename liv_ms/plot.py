'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=ungrouped-imports
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
from collections.abc import Iterable
import os

from matplotlib import collections
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(query, hits, out_dir='out'):
    '''Plot spectrum.'''
    query_lines = [[(x, 0), (x, y)] for x, y in query['spectrum']]
    query_col = ['green' for _ in query['spectrum']]

    # Make plot
    fig, axes = plt.subplots(len(hits), 1, sharex=True)

    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax, hit in zip(axes, hits):
        ax.axhline(y=0, color='k', linewidth=1)
        ax.margins(x=0, y=0)

        # Add 'peaks':
        ax.add_collection(
            collections.LineCollection(
                query_lines + [[(x, 0), (x, -y)] for x, y in hit['spectrum']],
                colors=query_col + ['red' for _ in hit['spectrum']],
                alpha=0.5))

        # Add (invisible) scatter points:
        ax.scatter(*zip(*query['spectrum']), s=0)
        ax.scatter(*zip(*hit['spectrum']), s=0)

        # Format and save:
        name = '_'.join([query['name'], hit['name'], '%.3f' % hit['score']])
        ax.set_title(name, fontsize=6)
        ax.set_xlabel('m/z', fontsize=6)
        ax.set_ylabel('I', fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=4)

    fig.tight_layout()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, query['name'] + '.png'), dpi=800)


def plot_loss(history, title, out_dir='out'):
    '''Plot training loss.'''
    try:
        plt.clf()

        plt.title(title)
        plt.plot(history.history['loss'], label='train')

        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='dev')

        plt.legend()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        plt.savefig(os.path.join(out_dir, '%s.png' % title), dpi=800)
    except AttributeError:
        pass


def plot_scatter(x, y, title, xlabel, ylabel, out_dir='out'):
    '''Scatter plot.'''
    plt.clf()

    # Flatten:
    if len(x.shape) > 1:
        x = x.flatten()

    if len(y.shape) > 1:
        y = y.flatten()

    # Set axes:
    max_val = np.ceil(max(max(x), max(y)))
    axes = plt.gca()
    axes.set_xlim([0, max_val])
    axes.set_ylim([0, max_val])
    plt.scatter(x, y, s=1)

    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    label = 'y = %.2f + %.2fx, R2 = %.2f' % (intercept, slope, r_value**2)

    plt.plot(x, [intercept + slope * xi for xi in x],
             label=label,
             linewidth=0.5)

    plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, title + '.png'), dpi=800)

    return label
