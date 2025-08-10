import pandas as pd
import spacy
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linprog
from pyemd import emd
import os
import csv
from typing import Dict, List


def analyze_tokenization(tokenizers_list, word_list, l1, l2, categories):
    """
    This function computes an analysis on how different tokenizers split words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param word_list: list of words
    :param l1: the first language
    :param l2: the second language
    :param algo: the name of the algorithm
    :param categories: tokenization cases
    :return: dictionary of {tokenization_case : [list of words]}
    """
    # init cases with value 0
    num_tokens_diff = {k: [] for k in categories}
    
    for word in word_list:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            res = t.tokenize(word)
            word_tokenization.append(res)
            num_tokens.append(len(res))
        # Same splits throughout all tokenizers
        if word_tokenization[0] == word_tokenization[2] and word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff["same_splits"].append(word)
        # Same tokenization between language1 and multilingual tokenizer
        elif word_tokenization[0] == word_tokenization[2]:
            num_tokens_diff[f"{l1}_t==multi_t"].append(word)
        # Same tokenization between language2 and multilingual tokenizer
        elif word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff[f"{l2}_t==multi_t"].append(word)
        # All different tokenization
        elif word_tokenization[0] != word_tokenization[1] and word_tokenization[0] != word_tokenization[2] and \
                word_tokenization[1] != word_tokenization[2]:
            num_tokens_diff["different_splits"].append(word)
        # Same tokenization between language1 and langauge2, but different from Multi tokenizer
        elif word_tokenization[0] == word_tokenization[1]:
            num_tokens_diff[f"{l1}_t=={l2}_t"].append(word)
    
    return num_tokens_diff



def extract_context(ff_words, text, window=5, max_context=100_000):
    """
    This function extracts all the appearances of a false friend word with its context with a certain window size.
    :param ff_words: a list of false friends words
    :param text: a string that contains an entire corpus
    :param window: a window size
    :param max_context: maximum number of contexts for a ff word
    :return: a dictionary {ff_word1: [context1, context2,...]}
    """
    # Split the text into words
    words = text.split()
    contexts = {}
    
    # Loop over each word to find
    for ff in ff_words:
        contexts[ff] = []
        num_contexts = 0
        # Iterate through text to find matches (case-insensitive)
        for i, word in enumerate(words):
            if word.lower() == ff:
                start = max(i - window, 0)
                end = min(i + window + 1, len(words))
                context = words[start:end]
                contexts[ff].append(' '.join(context))
                num_contexts += 1
            if num_contexts >= max_context:
                break
    return contexts


def get_pos_count(ff_words, context, nlp):
    """
    In this function, for each false friend word it counts how many times it was tagged in a certain POS. If the ff
    word does not have any context (i.e. did not appear in the corpus) then we tag it stand alone.
    :param ff_words: a list of false friend words
    :param context: the context in which the false friend word appears in a corpus
    :param nlp: spacy model
    :return: a dictionary {ff_word1: {POS1: x, POS2: y, POS3: z,...}}
    """
    
    ff_pos_counts = {f: dict() for f in ff_words}
    
    for ff in ff_words:
        for c in context[ff]:
            doc = nlp(c)
            for token in doc:
                if ff == token.text.lower():
                    ff_pos_counts[ff][token.pos_] = ff_pos_counts[ff].get(token.pos_, 0) + 1
                    break
    # if ff word not in the corpus (no context) then get the words stand alone POS
    for ff in ff_words:
        if len(context[ff]) == 0:
            doc = nlp(ff)
            for token in doc:
                ff_pos_counts[ff][token.pos_] = 1
    
    return ff_pos_counts


def get_pos_data(num_tokens_diff, l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, categories):
    """
    This function gets the POS data for each tokenization category for each language
    :param num_tokens_diff: dictionary {tokenization_case: [list of words]}
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :return: POS data for each tokenization category
    """
    spacy_dic = {"en": "en_core_web_sm", "fr": "fr_core_news_sm", "de": "de_core_news_sm", "it": "it_core_news_sm",
                 "ro": "ro_core_news_sm", "es": "es_core_news_sm", "se": "sv_core_news_sm"}
    
    nlp1 = spacy.load(spacy_dic[l1])
    nlp2 = spacy.load(spacy_dic[l2])
    # corpus_path1 = get_training_corpus_dir(training_corpus_dir1)
    # corpus_path2 = get_training_corpus_dir(training_corpus_dir2)
    with open(l1_training_corpus_dir, "r") as corpus1, open(l2_training_corpus_dir, "r") as corpus2:
        corpus_text1 = corpus1.read().replace("\n", " ")
        corpus_text2 = corpus2.read().replace("\n", " ")
    
    # getting all ff words
    ff_words = []
    for category in categories:
        for f in num_tokens_diff[category]:
            ff_words.append(f)
    # getting context of ff appearances
    context1 = extract_context(ff_words, corpus_text1)
    context2 = extract_context(ff_words, corpus_text2)
    
    # counting the POS for each word in the different languages
    ff_pos1_counts = get_pos_count(ff_words, context1, nlp1)
    ff_pos2_counts = get_pos_count(ff_words, context2, nlp2)
    
    # calculate distribution of POS for each category
    tokenization_category_pos1 = {category: dict() for category in categories}
    tokenization_category_pos2 = {category: dict() for category in categories}
    for ff in ff_words:
        # get most common POS for false friend
        pos1 = max(ff_pos1_counts[ff], key=ff_pos1_counts[ff].get)
        pos2 = max(ff_pos2_counts[ff], key=ff_pos2_counts[ff].get)
        for category in categories:
            if ff in num_tokens_diff[category]:
                tokenization_category_pos1[category][pos1] = tokenization_category_pos1[category].get(pos1, 0) + 1
                tokenization_category_pos2[category][pos2] = tokenization_category_pos2[category].get(pos2, 0) + 1
    
    return tokenization_category_pos1, tokenization_category_pos2


BASE_UPOS_ORDER: List[str] = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

POS_COLOR_MAP: Dict[str, str] = {
    "ADJ": "blue",
    "ADP": "orange",
    "ADV": "green",
    "AUX": "red",
    "CCONJ": "purple",
    "DET": "brown",
    "INTJ": "pink",
    "NOUN": "gray",
    "NUM": "olive",
    "PART": "cyan",
    "PRON": "lightblue",
    "PROPN": "gold",
    "PUNCT": "lightgreen",
    "SCONJ": "salmon",
    "SYM": "plum",
    "VERB": "tan",
    "X": "lightpink",
}

FALLBACK_COLORS = [
    "navy", "darkgreen", "goldenrod", "maroon", "indigo", "slateblue", "darkolivegreen",
    "peru", "firebrick", "orchid", "royalblue", "yellowgreen", "khaki", "crimson"
]

BASE_UPOS_ORDER = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

# Stable color mapping per POS tag (color names for readability & consistency across runs).
POS_COLOR_MAP = {
    "ADJ": "blue",
    "ADP": "orange",
    "ADV": "green",
    "AUX": "red",
    "CCONJ": "purple",
    "DET": "brown",
    "INTJ": "pink",
    "NOUN": "gray",
    "NUM": "olive",
    "PART": "cyan",
    "PRON": "lightblue",
    "PROPN": "gold",
    "PUNCT": "lightgreen",
    "SCONJ": "salmon",
    "SYM": "plum",
    "VERB": "tan",
    "X": "lightpink",
}

# Deterministic fallback colors for unseen tags (assigned in order if needed).
FALLBACK_COLORS = [
    "navy", "darkgreen", "goldenrod", "maroon", "indigo", "slateblue", "darkolivegreen",
    "peru", "firebrick", "orchid", "royalblue", "yellowgreen", "khaki", "crimson"
]

def pos_order_and_colors(all_pos):
    """Compute a stable POS order and a tag→color map.

    Parameters
    ----------
    all_pos : list of str
        All POS tags observed across both languages and all categories.

    Returns
    -------
    order : list of str
        POS order starting with BASE_UPOS_ORDER, followed by unseen tags alphabetically.
    color_map : dict
        Mapping from POS tag to a human-readable color name. Unseen tags use FALLBACK_COLORS.
    """
    extras = sorted([t for t in set(all_pos) if t not in BASE_UPOS_ORDER])
    order = BASE_UPOS_ORDER + extras
    color_map = dict(POS_COLOR_MAP)
    for i, tag in enumerate(extras):
        if tag not in color_map:
            color_map[tag] = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
    return order, color_map


def collect_all_pos(pos_dict_l1, pos_dict_l2):
    """Collect the union of POS tags present in L1 and L2 per-category dicts.
    list of str
        Sorted list of all POS tags observed.
    """
    s = set()
    for d in (pos_dict_l1, pos_dict_l2):
        for category in d:
            s.update(d[category].keys())
    return sorted(s)


def sorted_order_by_frequency(order, cat_counts_l1, cat_counts_l2):
    """Re-rank POS for a single category by total frequency (desc), with stable tiebreaks.

    Parameters
    ----------
    order : list of str
        Base POS order to use for stable tiebreaking.
    cat_counts_l1 : dict
        {POS: count} for the category in L1.
    cat_counts_l2 : dict
        {POS: count} for the category in L2.

    Returns
    -------
    list of str
        POS order sorted by (L1+L2) counts descending, ties by *order* index.
    """
    totals = {tag: cat_counts_l1.get(tag, 0) + cat_counts_l2.get(tag, 0) for tag in order}
    base_index = {tag: i for i, tag in enumerate(order)}
    return sorted(order, key=lambda t: (-totals.get(t, 0), base_index[t]))


def plot_pos_data(num_tokens_diff, l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, categories, algo, dir):
    

    # Get POS distributions per language per category using your provided pipeline
    tokenization_category_pos1, tokenization_category_pos2 = get_pos_data(num_tokens_diff, l1, l2, l1_training_corpus_dir,
                                                                          l2_training_corpus_dir, categories)

    # Collect the universe of POS tags observed across both languages
    all_pos = collect_all_pos(tokenization_category_pos1, tokenization_category_pos2)
    
    # Establish a stable overall order and color map for tags
    pos_order_base, color_map = pos_order_and_colors(all_pos)

    # Per-category plot
    for category in categories:
        counts_l1 = tokenization_category_pos1.get(category, {})
        counts_l2 = tokenization_category_pos2.get(category, {})

        # Re-rank by frequency within this category to improve readability, with stable ties
        pos_order = sorted_order_by_frequency(pos_order_base, counts_l1, counts_l2)

        # Build the bar positions
        x = list(range(len(pos_order)))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 6))

        # Bars for L1 and L2; same color (by POS), different hatches to distinguish languages
        for i, tag in enumerate(pos_order):
            c = color_map[tag]
            ax.bar(i - width/2, counts_l1.get(tag, 0), width, color=c, edgecolor="black", linewidth=0.6, hatch="")
            ax.bar(i + width/2, counts_l2.get(tag, 0), width, color=c, edgecolor="black", linewidth=0.6, hatch="//")

        # Axes & title
        ax.set_title(f"POS distribution: {category} (algo={algo}, {l1} vs {l2})")
        ax.set_xlabel("POS tag")
        ax.set_ylabel("Count of words (majority POS)")
        ax.set_xticks(x)
        ax.set_xticklabels(pos_order, rotation=45, ha="right")
        ax.margins(x=0.01)
        ax.grid(axis='y', linestyle=':', linewidth=0.6)

        # Legends
        from matplotlib.patches import Patch
        pos_legend_handles = [Patch(facecolor=color_map[tag], edgecolor='black', label=tag) for tag in pos_order]
        lang_legend_handles = [
            Patch(facecolor="lightgray", edgecolor='black', hatch="", label=f"{l1}"),
            Patch(facecolor="lightgray", edgecolor='black', hatch="//", label=f"{l2}"),
        ]
        legend1 = ax.legend(handles=lang_legend_handles, loc='upper right', title="Language")
        ax.add_artist(legend1)
        ax.legend(handles=pos_legend_handles, loc='upper left', ncol=2, title="POS")

        fig.tight_layout()

        # Save file
        safe_category = str(category).replace(" ", "_")
        fname = f"pos_{algo}_{l1}_vs_{l2}_{safe_category}.png"
        fig.savefig(os.path.join(dir, fname), dpi=200)
        plt.close(fig)


def chi_square_test(ff_num_tokens_diff, homographs_tokenization_cases, l1, l2, algo):
    """
    This function calculates the chi square test between the tokenization cases of False Friend words and the tokenization cases
    of words written the same in languages l1 and l2
    :param ff_data: the False Friends tokenization cases
    :param same_words_data: the same words across languages l1 and l2 tokenization cases
    :param l1: language 1
    :param l2: language 2
    :param algo: the name of the algorithm
    :return:
    """
    
    categories, ff_counts, same_words_counts = [], [], []
    for cat in ff_num_tokens_diff.keys():
        categories.append(cat)
        ff_counts.append(len(ff_num_tokens_diff[cat]))
        same_words_counts.append(len(homographs_tokenization_cases[cat]))
    
    # Create contingency table (2xN)
    contingency_table = np.array([ff_counts, same_words_counts])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print results
    print(f"Chi-Squared Test for {algo} on {l1}-{l2}")
    print(f"Categories: {categories}")
    print(f"False Friends counts: {ff_counts}")
    print(f"Same Words counts: {same_words_counts}")
    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant difference (p < 0.05)")
    else:
        print("❌ No statistically significant difference (p >= 0.05)")
    
    return chi2, dof, p_value, expected


def plot_tokenization_cases(num_tokens_diff, algo, l1, l2, categories, word_types, dir):
    """
    This function plots the tokenization cases
    :param num_tokens_diff: dictionary {tokenization_case: [list of words]}
    :param algo: algo name
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :param word_types: False Friends words or other list of words
    :param dir: directory to save graph
    :return:
    """
    
    plt.figure(figsize=(12, 14))
    x_axis = categories
    y_axis = [len(num_tokens_diff[key]) for key in x_axis]
    distribution = [f"{key}: {len(num_tokens_diff[key])}" for key in x_axis]
    num_words = sum(y_axis)
    fig_save_path = f"{dir}/{word_types}_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases\n{l1}, {l2}\nAlgo: {algo}\nNum words: {num_words}\nDistribution: {distribution}"
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=30, fontsize=13)
    plt.xlabel("Tokenization Splits", fontsize=15)
    plt.ylabel("Amount of Tokenization Case", fontsize=15)
    plt.title(title, fontsize=15)
    plt.savefig(fig_save_path)
    plt.show()


def plot_average_word_length(num_tokens_diff, algo, dir, l1, l2, categories):
    """
    This function plots the average word length and standard deviation of each tokenization category
    :param num_tokens_diff: a dictionary {tokenization_category: [list_of_words]}
    :param algo: the algo name
    :param dir: directory to save the figure
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :return:
    """
    means, stds = get_average_word_length(num_tokens_diff, categories)
    
    fig_save_path = f"{dir}/avg_word_length_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases - Average Word Length\nMean ± Std\n{l1}, {l2}\nAlgo: {algo}"
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(categories))
    plt.bar(x, means, yerr=stds, capsize=5, edgecolor='black')
    plt.xticks(x, categories, rotation=30, fontsize=12)
    plt.xlabel("Tokenization Case")
    plt.ylabel("Average Word Length")
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()


def get_average_word_length(num_tokens_diff, categories):
    """
    This function computes the average word length per category
    :param num_tokens_diff: dictionary of {tokenization_case : [list of words]}
    :param categories: tokenization cases
    :return: means and stds for each category
    """
    means = []
    stds = []
    for cat in categories:
        words = num_tokens_diff[cat]
        word_lengths = [len(w) for w in words]
        means.append(np.mean(word_lengths) if word_lengths else 0)
        stds.append(np.std(word_lengths) if word_lengths else 0)
    return means, stds


def plot_average_num_tokens(tokenizers_list, num_tokens_diff, algo, dir, l1, l2, categories):
    """
    This function plots the mean and standard deviation of tokens for each tokenization case
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param num_tokens_diff: a dictionary {category: [list of words]}
    :param algo: the algo name
    :param dir: directory to save the figure
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :return:
    """
    
    bar_means, bar_positions, bar_stds, bar_width, flat_labels = get_avg_num_tokens(algo, l1, l2, num_tokens_diff,
                                                                                    tokenizers_list, categories)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(bar_positions, bar_means, yerr=bar_stds, capsize=5, width=bar_width, edgecolor='black')
    plt.xticks(bar_positions, flat_labels, rotation=30)
    plt.ylabel("Number of tokens")
    plt.title(f"Tokenization Cases - Average Tokens\nMean ± Std\n{l1}, {l2}\nAlgo: {algo}")
    plt.tight_layout()
    
    # Save and show
    fig_save_path = f"{dir}/avg_tokens_{l1}_{l2}_{algo}.png"
    plt.savefig(fig_save_path)
    plt.show()


def get_avg_num_tokens(algo, l1, l2, num_tokens_diff, tokenizers_list, categories):
    """
    This function calculates the avg number of tokens per category
    :param algo: the algo name
    :param l1: language 1
    :param l2: language 2
    :param num_tokens_diff: a dictionary {tokenization_case: [list of words]}
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param categories: tokenization cases
    :return: data to plot the graph
    """
    num_tokens_case = {
        "same_splits": [],
        f"{l1}_t==multi_t": [[], []],
        f"{l2}_t==multi_t": [[], []],
        "different_splits": [[], [], []],
        f"{l1}_t=={l2}_t": [[], []],
    }
    for cat in categories:
        for w in num_tokens_diff[cat]:
            cur_tokenization = []
            for t in tokenizers_list:
                res = t.tokenize(w)
                cur_tokenization.append(res)
            if cat == "same_splits":
                num_tokens_case[cat].append(len(cur_tokenization[0]))
            elif cat == f"{l1}_t==multi_t":
                num_tokens_case[cat][0].append(len(cur_tokenization[0]))
                num_tokens_case[cat][1].append(len(cur_tokenization[1]))
            elif cat == f"{l2}_t==multi_t":
                num_tokens_case[cat][0].append(len(cur_tokenization[1]))
                num_tokens_case[cat][1].append(len(cur_tokenization[0]))
            elif cat == "different_splits":
                num_tokens_case[cat][0].append(len(cur_tokenization[0]))
                num_tokens_case[cat][1].append(len(cur_tokenization[1]))
                num_tokens_case[cat][2].append(len(cur_tokenization[2]))
            else:  # f"{l1}_t=={l2}_t"
                num_tokens_case[cat][0].append(len(cur_tokenization[0]))
                num_tokens_case[cat][1].append(len(cur_tokenization[2]))
    # Now compute means and stds for plotting
    all_group_labels = []
    bar_positions = []
    bar_means = []
    bar_stds = []
    current_x = 0
    bar_width = 0.25
    group_spacing = 1  # space between groups
    for cat in categories:
        data = num_tokens_case[cat]
        
        if cat != "same_splits":  # grouped case
            n = len(data)
            for i in range(n):
                values = data[i]
                bar_means.append(np.mean(values) if values else 0)
                bar_stds.append(np.std(values) if values else 0)
                bar_positions.append(current_x + (i - (n - 1) / 2) * bar_width)
            # Define appropriate labels per group
            if cat == f"{l1}_t==multi_t":
                all_group_labels.append([f"{l1}_t==multi_t", f"{l2}_t"])
            elif cat == f"{l2}_t==multi_t":
                all_group_labels.append([f"{l2}_t==multi_t", f"{l1}_t"])
            elif cat == "different_splits":
                all_group_labels.append([f"{l1}_t", f"{l2}_t", "multi_t"])
            elif cat == f"{l1}_t=={l2}_t":
                all_group_labels.append([f"{l1}_t=={l2}_t", "multi_t"])
            current_x += group_spacing
        else:  # single bar case
            values = data
            bar_means.append(np.mean(values) if values else 0)
            bar_stds.append(np.std(values) if values else 0)
            bar_positions.append(current_x)
            all_group_labels.append(["same_splits"])
            current_x += group_spacing
    # Flatten labels for x-axis
    flat_labels = [label for group in all_group_labels for label in group]
    return bar_means, bar_positions, bar_stds, bar_width, flat_labels


def plot_frequency_comparison(num_tokens_diff, algo, dir, l1, l2, word_freqs1, word_freqs2, categories):
    """
    This function plots a graph of the mean and standard deviation of the False Friends frequencies in the training corpus
    of the l1 and l2 languages
    :param num_tokens_diff: a dictionary {tokenization_category: [list_of_words]}
    :param algo: the algo name
    :param dir: directory to save the figure
    :param word_frequencies: word frequencies in the training corpus
    :param l1: language 1
    :param l2: language 2
    :param categories: list of categories
    :return:
    """
    
    mean1, mean2 = [], []
    lower1, upper1 = [], []
    lower2, upper2 = [], []
    category_freq1 = dict()
    category_freq2 = dict()
    words_in_corpus = 0
    
    # Collect frequency data
    for category in categories:
        category_freq1[category] = []
        category_freq2[category] = []
        words = num_tokens_diff[category]
        for word in words:
            if word in word_freqs1 and word in word_freqs2:
                category_freq1[category].append(word_freqs1[word])
                category_freq2[category].append(word_freqs2[word])
                words_in_corpus += 1
    
    # Compute mean and std
    for category in categories:
        freqs1 = category_freq1[category]
        freqs2 = category_freq2[category]
        
        m1 = np.mean(freqs1) if freqs1 else 0
        s1 = np.std(freqs1) if freqs1 else 0
        m2 = np.mean(freqs2) if freqs2 else 0
        s2 = np.std(freqs2) if freqs2 else 0
        
        mean1.append(m1)
        mean2.append(m2)
        
        # Asymmetric error bars
        lower1.append(min(s1, m1) if m1 > 0 else 0)
        upper1.append(s1)
        
        lower2.append(min(s2, m2) if m2 > 0 else 0)
        upper2.append(s2)
    
    # Plot
    fig_save_path = f"{dir}/frequencies_{l1}_{l2}_{algo}.png"
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(categories))
    plt.yscale("log")
    plt.bar(x - bar_width / 2, mean1, yerr=[lower1, upper1], capsize=5, width=bar_width, color='lightblue',
            label=f"{l1}")
    plt.bar(x + bar_width / 2, mean2, yerr=[lower2, upper2], capsize=5, width=bar_width, color='palegreen',
            label=f"{l2}")
    plt.xticks(x, categories, rotation=45)
    plt.ylabel("Frequency")
    plt.title(
        f"Tokenization Case Frequencies\nMean ± Std\n{l1}_{l2}_{algo}\nFalse Friends in Corpus: {words_in_corpus}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()


def missing_ff_in_corpus(ff_data, word_frequencies, dir):
    # TODO: not using for now
    """
    Creates a file of the missing False Friend words in the training corpus
    :param ff_data: the false friends data of a specific language
    :param word_frequencies: word frequencies of a specific language
    :param dir: the directory to save the .txt file
    :return:
    """
    
    with open(f"{dir}/missing_words.txt", 'w', encoding='utf-8') as f:
        for word in ff_data:
            if word not in word_frequencies.keys():
                f.write(f"{word}\n")


def write_tokenization_split(tokenizers, ff_data, l1, l2, algo, dir):
    """
    Writes the tokenization splits of different tokenizers to a .txt file
    :param tokenizers: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param ff_data: the ff data
    :param l1: language 1 (english)
    :param l2: language 2
    :param algo: the algorithm used
    :param dir: path to save .txt file
    :return:
    """
    with open(f"{dir}/{algo}.txt", 'w', encoding='utf-8') as f:
        f.write(f"{l1}_tokenizer, {l2}_tokenizer, {l1}_{l2}_tokenizer\n")
        for ff in ff_data:
            to_write = f""
            for t in tokenizers:
                to_write += f"{t.tokenize(ff)}"
            to_write += "\n"
            f.write(to_write)


def earth_movers_dist(categories, l1, l2, source, target, track_target=None):
    """
    Computes the Earth Movers Distance metric between to distributions. Also able to track how much earth was moved
     to a specific target (track_target)
    :param categories: the categories
    :param l1: English
    :param l2: other language
    :param source: source distribution
    :param target: target distribution
    :param track_target: target category to track
    :return: emd, moved
    """
    s = np.array([source[c] for c in categories], dtype=np.float64)
    t = np.array([target[c] for c in categories], dtype=np.float64)
    
    # Normalizing
    s /= s.sum()
    t /= t.sum()
    
    n = len(s)
    
    # Create distance matrix
    D = np.array([[dist(l1, l2, c1, c2) for c1 in categories] for c2 in categories], dtype=np.float64)
    # we are trying to minimize c.T@x where x is the solution for the linear program. So, c is the cost
    c = D.flatten()
    
    # Creating equality constraints
    A_eq = []
    b_eq = []
    
    # Supply constraints
    # [[ f00, f01, f02 ],
    # [ f10, f11, f12 ], ---> [f00, f01, f02, f10, f11, f12, f20, f21, f22]
    # [ f20, f21, f22 ]]
    # We add the row constraints. A_eq[i][j] for all j must sum to s[i]. This means we cannot move more "dirt" than we have in s[i]
    for i in range(n):
        matrix = np.zeros((n, n))
        matrix[i, :] = 1
        A_eq.append(matrix.flatten())
        b_eq.append(s[i])
    
    # We add more constraints. A_eq[i][j] for all i must sum to t[j]. This means we want to get exactly the amount of "dirt" at t[j]
    for j in range(n):
        matrix = np.zeros((n, n))
        matrix[:, j] = 1  # All rows in column j (incoming flows)
        A_eq.append(matrix.flatten())
        b_eq.append(t[j])
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    flow_matrix = res.x.reshape((n, n))
    # Elementwise multiplication
    emd = np.sum(flow_matrix * D)
    if track_target is not None and track_target in categories:
        j = categories.index(track_target)
        moved = {categories[i]: flow_matrix[i][j] for i in range(n)}
        return emd, moved
    return emd


def dist(l1, l2, source, target):
    """
    The distance function for Earth Movers target function
    :param l1: English
    :param l2: other language
    :param source: source category
    :param target: target category
    :return:
    """
    d = {
        "same_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "different_splits": 2,
                        "same_splits": 0},
        "different_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "same_splits": 2,
                             "different_splits": 0},
        f"{l1}_t==multi_t": {f"same_splits": 1, f"{l2}_t==multi_t": 0.5, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l1}_t==multi_t": 0},
        f"{l2}_t==multi_t": {f"{l1}_t==multi_t": 0.5, f"same_splits": 1, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l2}_t==multi_t": 0},
        f"{l1}_t=={l2}_t": {f"{l1}_t==multi_t": 0.7, f"{l2}_t==multi_t": 0.7, f"same_splits": 1, "different_splits": 1,
                            f"{l1}_t=={l2}_t": 0}
    }
    
    return d[source][target]

def emd2(source, target, l1, l2, categories, algo1, algo2):
    """
    Earth Movers Distance function that uses pyemd. Gave the same result as linear programming
    :param source: source distribution
    :param target: target distribution
    :param l1: English
    :param l2: other language
    :param categories: categories
    :param algo1: algo name 1
    :param algo2: algo name 2
    :return: emd
    """
    s = np.array([source[c] for c in categories], dtype=np.float64)
    t = np.array([target[c] for c in categories], dtype=np.float64)
    D = np.array([[dist(l1, l2, c1, c2) for c1 in categories] for c2 in categories], dtype=np.float64)

    
    # Normalizing
    s /= s.sum()
    t /= t.sum()
    ans = emd(s, t, D)
    print(f"Earth Mover Distance for {l1}-{l2}")
    print(f"Categories: {categories}")
    print(f"Source Distribution {algo1}: {source}")
    print(f"Target Distribution {algo2}: {target}")
    print(f"Earth Mover Distance: {ans}")

def get_avg_chars_per_token(tokenizer):
    """
    Calculates the average characters per token for tokenizer vocabulary
    :param tokenizer: the tokenizer object
    :return: cpt
    """
    vocab = tokenizer.get_vocab()
    num_chars = sum([len(v) for v in vocab])
    return num_chars / len(vocab)

def get_token_length_distribution(tokenizer):
    """
    Returns the token length distribution oh the tokens in the tokenizer vocabulary
    :param tokenizer: the tokenizer object
    :return:
    """
    vocab = tokenizer.get_vocab()
    distribution = dict()
    for v in vocab:
        distribution[len(v)] = distribution.get(len(v), 0) + 1
    for k, v in distribution.items():
        distribution[k] = v / len(vocab)
    sorted_dis = {key: distribution[key] for key in sorted(distribution.keys())}
    return sorted_dis


def words_moved_to_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from num_tokens_diff1 to a certain category in num_tokens_diff2
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2:tokenization cases 2
    :param categories:categories
    :param target: which words moved to target in tokenization cases 2
    :return: words moved to target
    """
    words_moved = {c:[] for c in categories}
    for c, words in num_tokens_diff1.items():
        added = set(num_tokens_diff1[c]) & set(num_tokens_diff2[target])
        for w in added:
            words_moved[c].append(w)
    return words_moved

def words_removed_from_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from target in num_tokens_diff2 to other categories in num_tokens_diff1
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2: tokenization cases 2
    :param categories: categories
    :param target: which words moved out from target in tokenization cases 2
    :return: words moved out from target
    """
    words_moved = {c: [] for c in categories if c != target}
    for w in num_tokens_diff1[target]:
        if w not in set(num_tokens_diff2[target]):
            for c in words_moved.keys():
                if w in set(num_tokens_diff2[c]):
                    words_moved[c].append(w)
    return words_moved


# INTRINSIC ANALYSIS FUNCTIONS

def ff_intrinsic_analysis(l1, l2, ex):
    ff_data_path_dict = {"de": "C:\\Users\\halor\\Desktop\\Masters_degree\\Melel\\thesisV02\\ff_data\\de_ff.csv",
                         "es": "C:\\Users\\halor\\Desktop\\Masters_degree\\Melel\\thesisV02\\ff_data\\spanish_ff.csv",
                         "se": "C:\\Users\\halor\\Desktop\\Masters_degree\\Melel\\thesisV02\\ff_data\\swedish_ff.csv"}
    if l2 not in ff_data_path_dict.keys():
        return
    with open(ff_data_path_dict[l2], 'r', encoding='utf-8') as f:
        ff_data = list(csv.DictReader(f))
    en_tokenizer = ex.l1_tokenizer
    l2_tokenizer = ex.l2_tokenizer
    en_l2_tokenizer  = ex.l1_l2_tokenizer
    path = f"{ex.analysis_dir}/intrinsic_analysis"
    os.makedirs(path, exist_ok=True)
    
    spacy_dic = {"en": "en_core_web_sm", "fr": "fr_core_news_sm", "de": "de_core_news_sm", "it": "it_core_news_sm",
                 "ro": "ro_core_news_sm", "es": "es_core_news_sm", "se": "sv_core_news_sm"}
    
    nlp1 = spacy.load(spacy_dic[l1])
    nlp2 = spacy.load(spacy_dic[l2])
    
        
    with open(f"{path}/{ex.algo_name}_intrinsic.txt", 'w', encoding='utf-8') as f:
        for ff_row in ff_data:
            ff_word = ff_row["False Friend"]
            correct_translation = ff_row["Correct Translation"]
            wrong_translation = ff_row["Wrong Translation"]
            en_sentences = [ff_row["en sentence1"], ff_row["en sentence2"]]
            l2_sentences = [ff_row[f"{l2} sentence1"], ff_row[f"{l2} sentence2"]]
            f.write("##########################################################################################################################\n")
            f.write(f"Analysis for False Friend word: {ff_word}\n")
            f.write(f"Correct Translation: {correct_translation}\n")
            f.write(f"Wrong Translation: {wrong_translation}\n\n")
            f.write(f"'{ff_word}' in English context:\n")
            for i in range(len(en_sentences)):
                f.write(f"Sentence {i + 1}:\n")
                f.write(f"{en_sentences[i]}\n")
                for w in en_sentences[i].split(" "):
                    f.write(f"{en_tokenizer.tokenize(w)} ")
                f.write("\n")
                doc = nlp1(en_sentences[i])
                for token in doc:
                    f.write(f"{(token.text, token.pos_)}, ")
                f.write("\n")
            f.write(f"'{ff_word}' in {l2} context:\n")
            for j in range(len(l2_sentences)):
                f.write(f"Sentence {j + 1}:\n")
                f.write(f"{l2_sentences[j]}\n")
                for w in l2_sentences[j].split(" "):
                    f.write(f"{l2_tokenizer.tokenize(w)} ")
                f.write("\n")
                doc = nlp2(l2_sentences[j])
                for token in doc:
                    f.write(f"{(token.text, token.pos_)}, ")
                f.write("\n")
            f.write("Multilingual tokenization: English\n")
            for i in range(len(en_sentences)):
                for w in en_sentences[i].split(" "):
                    f.write(f"{en_l2_tokenizer.tokenize(w)} ")
                f.write("\n")
            f.write(f"Multilingual tokenization: {l2}\n")
            for i in range(len(l2_sentences)):
                for w in l2_sentences[i].split(" "):
                    f.write(f"{en_l2_tokenizer.tokenize(w)} ")
                f.write("\n")
            f.write("##########################################################################################################################\n")


                
            
    
    
    
# l1 = "en"
# l2 = "de"
# categories = [f"{l1}_t==multi_t", f"{l2}_t==multi_t", f"{l1}_t=={l2}_t", "same_splits", "different_splits"]
# s = {f"{l1}_t==multi_t": 0, f"{l2}_t==multi_t": 0, f"{l1}_t=={l2}_t": 0, "different_splits": 10, "same_splits": 0}
# t = {f"{l1}_t==multi_t": 0, f"{l2}_t==multi_t": 0, f"{l1}_t=={l2}_t": 0, "different_splits": 0, "same_splits": 10}
# emd2(s, t, l1, l2, categories, "BPE", "BPE_SAGE")
# earth_movers_dist(categories, l1, l2, "BPE", "BPE_SAGE", s, t, track_target="same_splits")