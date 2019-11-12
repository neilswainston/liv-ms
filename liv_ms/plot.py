'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order
from collections.abc import Iterable
import os

from matplotlib import collections

import matplotlib.pyplot as plt


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
