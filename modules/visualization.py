import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

from config import seg_profiles, visualization_conf, Language
from typing import List, Dict, Any


def plot_model_sizes(vocab_size: List[int], file_size: List[int], title: str="", save_path: str=None):
    """ Plot a bar plot of the total file size vs. the vocabulary size
    :param vocab_size: vocabulary sizes
    :param file_size: file sizes
    :param title: title of the plot
    :param save_path: path to save the plot
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0,0,1,1])

    rects1 = ax.bar(vocab_size, file_size, width=70)
    ax.set_ylim([3900000, 4400000])
    ax.set_title(title)
    ax.set_ylabel("Total file size")
    ax.set_xlabel("Vocabulary size")
    ax.set_xticks(vocab_size)
    ax.bar_label(rects1, padding=3, rotation='vertical', fmt='%d')

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    plt.show()

    if save_path is not None:
        fig.savefig(save_path)

def plot_grid_results(LANG: Language, grid_results:Dict, save_path:str=None):
    """ Plot the perplexity evolution of the three models for 
    the different values of each parameter
    :param LANG: language 
    :param grid_results: results of the grid search
    :param save_path: save the plot as .png or not
    """
    profiles = seg_profiles[LANG.name]

    # Extract perplexity evolutions
    hidden_choices = set()
    bptt_choices = set()
    n_class_choices = set()

    hidden_evolution = dict()
    bptt_evolution = dict()
    n_class_evolution = dict()

    for seg in profiles:
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

    # Extract maximum perplexity
    max_perplexity = max(grid_results.values())


    # Plot the evolution for each hyper-parameter
    fig, ax = plt.subplots(ncols=3, figsize=(16,5))
    fig.subplots_adjust(hspace=.3)
    legend = [seg.id for seg in profiles]

    for seg in profiles:
        ax[0].plot(hidden_choices, hidden_evolution[seg.id])

    ax[0].set_xlabel("# of hidden neurons")
    ax[0].set_xticks(hidden_choices)
    ax[0].set_ylabel("Perplexity")
    ax[0].set_ylim(0, max_perplexity)
    ax[0].legend(legend)


    for seg in profiles:
        ax[1].plot(bptt_choices, bptt_evolution[seg.id])
    ax[1].set_xlabel("bptt")
    ax[1].set_xticks(bptt_choices)
    ax[1].set_ylabel("Perplexity")
    ax[1].set_ylim(0, max_perplexity)
    ax[1].legend(legend)

    for seg in profiles:
        ax[2].plot(n_class_choices, n_class_evolution[seg.id])
    ax[2].set_xlabel("class")
    ax[2].set_xticks(n_class_choices)
    ax[2].set_xticklabels(n_class_choices, rotation=90)
    ax[2].set_ylabel("Perplexity")
    ax[2].set_ylim(0, max_perplexity)
    ax[2].legend(legend)

    # Save figure if save_path is provided
    if save_path is not None:
        fig.savefig(save_path)

def plot_oov_rates(oov_rates, save_path=None):
    """
    Plot the OOV rates of difference models with varying generated text size.

    Parameters:
        oov_rates: a list of list. Each list corresponds to a model and contains
    the oov rates of that model with varying generated text size.
        save_path: a string containing the path to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for id, oov_list in enumerate(oov_rates, start=1):
        n_gen = [10**(i+1) for i in range(len(oov_list))]
        ax.plot(n_gen, oov_list, label=f's{id}')
        ax.set_title('OOV rate changes with size of the generated text')
        ax.set_xlabel('generated tokens')
        ax.set_ylabel('OOV rate (%)')
        ax.set_xscale('log')

    plt.legend()
    plt.show()

    # Save figure if save_path is provided
    if save_path is not None:
        fig.savefig(save_path)