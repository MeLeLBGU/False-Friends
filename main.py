import torch
import sys
import json
import random
from statistical_analysis import *
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
    
    for l2, exp_list in experiments.items():
        for ex in exp_list:
            graphs_path = f"{ex.analysis_dir}/graphs"
            tokenization_path = f"{ex.analysis_dir}/tokenization"
            categories = get_categories(ex)
            ff_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2,
                                                         categories)
            same_words_tokenization_cases = analyze_tokenization(ex.get_tokenizers_list(),
                                                                 ex.get_same_words_across_languages(),
                                                                 ex.l1, ex.l2, categories)
            plot_tokenization_cases(ff_tokenization_cases, ex.algo_name, ex.l1, ex.l2, categories, "ff",
                                    graphs_path)
            write_tokenization_split(ex.get_tokenizers_list(), ex.get_ff_words(), ex.l1, ex.l2, ex.algo_name,
                                     tokenization_path)
            plot_average_word_length(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2, categories)
            plot_average_num_tokens(ex.get_tokenizers_list(), ff_tokenization_cases, ex.algo_name, graphs_path,
                                    ex.l1, ex.l2, categories)
            plot_frequency_comparison(ff_tokenization_cases, ex.algo_name, graphs_path, ex.l1, ex.l2,
                                      ex.get_corpus_words(ex.l1), ex.get_corpus_words(ex.l2), categories)
            plot_pos_data(ff_tokenization_cases, ex.l1, ex.l2, ex.l1_training_corpus_dir, ex.l2_training_corpus_dir,categories, ex.algo_name, graphs_path)
            chi_square_test(ff_tokenization_cases, same_words_tokenization_cases, ex.l1, ex.l2, ex.algo_name)
            print(
                "#########################################################################################################")





