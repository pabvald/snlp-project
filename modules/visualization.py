import pickle
import matplotlib.pyplot as plt 

from config import seg_profiles, visualization_conf, figures_path
from typing import List, Dict, Any

def plot_grid_results(grid_results:Dict, save_path:str=None):
    """ Plot the perplexity evolution of the three models for 
    the different values of each parameter
    :param grid_results: results of the grid search
    :param save_path: save the plot as .png or not
    """

    # Extract perplexity evolutions
    hidden_choices = set()
    bptt_choices = set()
    n_class_choices = set()

    hidden_evolution = dict()
    bptt_evolution = dict()
    n_class_evolution = dict()

    for seg in seg_profiles:
        hidden_evolution[seg.id] = []
        bptt_evolution[seg.id] = []
        n_class_evolution[seg.id] = []
        
        for (_, seg_id, hidden, bptt, n_class), perplexity in grid_results.items():
            hidden_choices.add(hidden)
            bptt_choices.add(bptt)
            n_class_choices.add(n_class)

            if (seg.id == seg_id) and \
                (visualization_conf.bptt == bptt) and \
                (visualization_conf.n_class == n_class):
                    hidden_evolution[seg.id].append(perplexity)

            if (seg.id == seg_id) and \
                (visualization_conf.hidden == hidden) and \
                (visualization_conf.n_class == n_class):
                    bptt_evolution[seg.id].append(perplexity)

            if (seg.id == seg_id) and \
                (visualization_conf.hidden == hidden) and \
                (visualization_conf.bptt == bptt):
                    n_class_evolution[seg.id].append(perplexity)

    hidden_choices = sorted(list(hidden_choices))
    bptt_choices = sorted(list(bptt_choices))
    n_class_choices = sorted(list(n_class_choices))


    # Plot the evolution for each hyper-parameter
    fig, ax = plt.subplots(3, figsize=(5,15))
    fig.subplots_adjust(hspace=.3)
    legend = [seg.id for seg in seg_profiles]

    for seg in seg_profiles:
        ax[0].plot(hidden_choices, hidden_evolution[seg.id])

    ax[0].set_xlabel("# of hidden neurons")
    ax[0].set_xticks(hidden_choices)
    ax[0].set_ylabel("Perplexity")
    ax[0].legend(legend)


    for seg in seg_profiles:
        ax[1].plot(bptt_choices, bptt_evolution[seg.id])
    ax[1].set_xlabel("bptt")
    ax[1].set_xticks(bptt_choices)
    ax[1].set_ylabel("Perplexity")
    ax[1].legend(legend)

    for seg in seg_profiles:
        ax[2].plot(n_class_choices, n_class_evolution[seg.id])
    ax[2].set_xlabel("class")
    ax[2].set_xticks(n_class_choices)
    ax[2].set_ylabel("Perplexity")
    ax[2].legend(legend)

    # Save figure if save_path is provided
    if save_path is not None:
        fig.savefig(save_path)