import torch
import sys
import json
import random
from statistical_analysis import *
from data_preprocess_scripts.find_ff_all_words_all_languages import get_same_words_across_languages
from Experiment import Experiment
from MySageTokenizer import MySageTokenizer
from SPTokenizer import SPTokenizer
import os


def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def fix_path(path):
    return path.strip().replace("\\", "/")


def create_multi_text_file(path1, path2, file_name, num_rows=300_000, seed=42):
    """
    Creates a .txt file that combines two different text files by randomly sampling half of the lines
    from each input file using a specific random seed.

    :param path1: Path to file of first language
    :param path2: Path to file of second language
    :param file_name: Name of the combined output file
    :param num_rows: Total number of rows in the output file (half from each input)
    :param seed: Random seed for reproducibility
    """
    rows_from_each = num_rows // 2
    
    with open(path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    
    random.seed(seed)
    sampled1 = random.sample(lines1, rows_from_each)
    random.seed(seed + 1)
    sampled2 = random.sample(lines2, rows_from_each)
    
    with open(file_name, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sampled1 + sampled2)


def init_experiments(data, l1_tokenizers):
    experiments = dict()
    l1_data = data["l1"]
    l1 = l1_data["language"]
    l1_training_corpus_dir = fix_path(l1_data["training_data"])
    l1_words_dir = fix_path(l1_data["words"])
    l2_experiments = data["l2"]
    for l2_data in l2_experiments:
        l2 = l2_data["language"]
        experiments[l2] = []
        l2_training_corpus_dir = fix_path(l2_data["training_data"])
        l2_words_dir = fix_path(l2_data["words"])
        ff_words_path = fix_path(l2_data["ff"])
        l1_l2_training_corpus_dir = f"./training_data/{l2}/{l1}_{l2}_corpus.txt"
        create_multi_text_file(l1_training_corpus_dir, l2_training_corpus_dir, l1_l2_training_corpus_dir)
        for algo in data["algos"]:
            l1_tokenizer = l1_tokenizers[algo]
            if "SAGE" in algo:
                cur_exp = Experiment(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, l1_words_dir, l2_words_dir,
                                     l1_l2_training_corpus_dir, algo, data["vocab_size"], ff_words_path, l1_tokenizer,
                                     schedule=data["schedule"], initial_vocab_size=data["initial_vocab_size"])
            else:
                cur_exp = Experiment(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, l1_words_dir, l2_words_dir,
                                     l1_l2_training_corpus_dir, algo, data["vocab_size"], ff_words_path, l1_tokenizer)
            experiments[l2].append(cur_exp)
    
    return experiments


def train_l1_tokenizers(data):
    algos = data['algos']
    vocab_size = data['vocab_size']
    initial_vocab_size = data['initial_vocab_size']
    schedule = data['schedule']
    l1_data = data['l1']
    l1 = l1_data["language"]
    l1_tokenizers = dict()
    training_corpus_dir = fix_path(l1_data['training_data'])
    for algo in algos:
        if "SAGE" in algo:
            dir = f"./results/{l1}_{algo}_{vocab_size}"
            os.makedirs(dir, exist_ok=True)
            print(f"created directory {dir}")
            tokenizer = MySageTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo, schedule,
                                        initial_vocab_size)
        else:
            tokenizer = SPTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo)
        tokenizer.train_tokenizer()
        l1_tokenizers[algo] = tokenizer
    return l1_tokenizers


def start_experiments(experiments):
    for l2, exp_list in experiments.items():
        for exp in exp_list:
            exp.start_experiment()


def save_experiments(experiments):
    for l2, exp_list in experiments.items():
        for exp in exp_list:
            exp.save_experiment()


def load_experiments(path):
    experiments = dict()
    for l2 in os.listdir(path):
        experiments[l2] = []
        for pickle_file in os.listdir(f"{path}/{l2}"):
            experiments[l2].append(Experiment.load_experiment(f"{path}/{l2}/{pickle_file}"))
    return experiments


def get_categories(experiment):
    l1 = experiment.l1
    l2 = experiment.l2
    categories = [f"{l1}_t==multi_t", f"{l2}_t==multi_t", f"{l1}_t=={l2}_t", "same_splits", "different_splits"]
    return categories

def get_ex_couple(exp_list, algo):
    reg_ex, sage_ex = None, None
    for ex in exp_list:
        if ex.algo_name == algo:
            reg_ex = ex
        elif ex.algo_name == f"{algo}_SAGE":
            sage_ex = ex
    return reg_ex, sage_ex


def compare_trials(exp_list, baseline_algo, track_target):
    ex_reg, ex_sage = get_ex_couple(exp_list, baseline_algo)
    path = f"{ex_reg.analysis_dir}/tokenization/{baseline_algo}_baseline_vs_SAGE.txt"
    categories = get_categories(ex_reg)
    homographs = get_same_words_across_languages(ex_reg.l1, ex_reg.l2)
    tokenization_cases = analyze_tokenization(ex_reg.get_tokenizers_list(), homographs, ex_reg.l1, ex_reg.l2, categories)
    sage_tokenization_cases = analyze_tokenization(ex_sage.get_tokenizers_list(), homographs, ex_sage.l1, ex_sage.l2, categories)
    distribution1 = {c: len(tokenization_cases[c]) for c in categories}
    sage_distributions = {c: len(sage_tokenization_cases[c]) for c in categories}
    emd, moved = earth_movers_dist(categories, ex_reg.l1, ex_reg.l2, distribution1, sage_distributions, track_target)
    # Percent moved to tracked target
    total_mass_of_target = sum([moved[c] for c in categories])
    percent_moved = {c: moved[c]/total_mass_of_target for c in categories}
    avg_token_len1, avg_token_len2, avg_token_len3 = get_avg_token_length(homographs, ex_reg.l1_tokenizer), get_avg_token_length(homographs, ex_reg.l2_tokenizer), get_avg_token_length(homographs, ex_reg.l1_l2_tokenizer)
    sage_avg_token_len1, sage_avg_token_len2, sage_avg_token_len3 = get_avg_token_length(homographs, ex_sage.l1_tokenizer), get_avg_token_length(homographs, ex_sage.l2_tokenizer), get_avg_token_length(homographs, ex_sage.l1_l2_tokenizer)
    added  = words_moved_to_target(tokenization_cases, sage_tokenization_cases, categories, track_target)
    removed = words_removed_from_target(tokenization_cases, sage_tokenization_cases, categories, track_target)
    with open(path, "w", encoding="utf-8") as f:
        title = (f"Homographs across languages {ex_reg.l1} and {ex_reg.l2} - Baseline tokenizer: {baseline_algo}\n"
                 f"Difference between experiment {ex_reg.l1}_{ex_reg.algo_name}, {ex_reg.l2}_{ex_reg.algo_name}, multilingual_{ex_reg.algo_name} AND experiment "
                          f"{ex_sage.l1}_{ex_sage.algo_name}, {ex_sage.l2}_{ex_sage.algo_name}, multilingual_{ex_sage.algo_name}\n")
        distributions = f"{ex_reg.algo_name}: {distribution1}\n{ex_sage.algo_name}: {sage_distributions}\n"
        earth_movers = f"Earth Movers Distance: {emd}\nMass moved to {track_target}: {moved}\n"
        earth_movers2_percent = f"Percent of mass from source distribution to target {track_target}: {percent_moved}\n"
        avg_token_len_line1 = f"Average Token Length {ex_reg.algo_name}: {ex_reg.l1}_tokenizer:{avg_token_len1}, {ex_reg.l2}_tokenizer: {avg_token_len2}, multilingual_tokenizer: {avg_token_len3}\n"
        avg_token_len_line2 = f"Average Token Length {ex_sage.algo_name}: {ex_sage.l1}_tokenizer:{sage_avg_token_len1}, {ex_sage.l2}_tokenizer: {sage_avg_token_len2}, multilingual_tokenizer: {sage_avg_token_len3}\n"
        f.write(title)
        f.write(distributions)
        f.write(earth_movers)
        f.write(earth_movers2_percent)
        f.write(avg_token_len_line1)
        f.write(avg_token_len_line2)
        for c, words in added.items():
            f.write(f"Words added from source {c} to target {track_target}: {len(words)}\n")
        f.write("\n")
        for c, words in removed.items():
            f.write(f"Words removed from source {track_target} to target {c}: {len(words)}\n")
        f.write("\n")
        for c, words in added.items():
            f.write(f"Words added from source {c} to target {track_target}\n")
            for w in words:
                f.write(f'{w}\n')
            f.write("###################################################################################################################################################\n")
        f.write("\n")
        for c, words in removed.items():
            f.write(f"Words removed from source {track_target} to target {c}\n")
            for w in words:
                f.write(f'{w}\n')
            f.write("###################################################################################################################################################\n")
        
    
    


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train, args_path = sys.argv[1:]
    print(sys.argv[1:])
    data = parse_args(fix_path(args_path))
    if train == "True":
        l1_tokenizers = train_l1_tokenizers(data)
        print("Finished training l1 tokenizers")
        experiments = init_experiments(data, l1_tokenizers)
        print("Finished creating experiments")
        start_experiments(experiments)
        print("Finished starting experiments")
        save_experiments(experiments)
        print("Finished saving experiments")
    else:
        vocab_size = str(data["vocab_size"])
        experiments = load_experiments(f"./experiments/{vocab_size}")
        print("Finished loading experiments")
    
    # for l2, exp_list in experiments.items():
    #     for ex in exp_list:
    #         graphs_path = f"{ex.analysis_dir}/graphs"
    #         tokenization_path = f"{ex.analysis_dir}/tokenization"
    #         categories = get_categories(ex)
    #         ff_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2,
    #                                                      categories)
    #         homographs = get_same_words_across_languages(ex.l1, ex.l2)
    #         homographs_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(),
    #                                                              homographs,
    #                                                              ex.l1, ex.l2, categories)
            # plot_tokenization_cases(ff_tokenization_cases, ex.algo_name, ex.l1, ex.l2, categories, "ff",
            #                         graphs_path)
            # write_tokenization_split(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2, ex.algo_name,
            #                          tokenization_path)
            # plot_average_word_length(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2, categories)
            # plot_average_num_tokens(ex.get_tokenizers_list(), ff_tokenization_cases, ex.algo_name, graphs_path,
            #                         ex.l1, ex.l2, categories)
            # plot_frequency_comparison(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2,
            #                           ex.get_corpus_words(ex.l1), ex.get_corpus_words(ex.l2), categories)
            # plot_pos_data(ff_tokenization_cases, ex.l1, ex.l2, ex.l1_training_corpus_dir, ex.l2_training_corpus_dir,categories, ex.algo_name, graphs_path)
            # chi_square_test(ff_tokenization_cases, homographs_tokenization_cases, ex.l1, ex.l2, ex.algo_name)
            # print("#########################################################################################################")
        # compare_trials(exp_list, "BPE", "same_splits")
        # compare_trials(exp_list, "UNI", "same_splits")
        #





