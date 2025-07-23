import os
import pickle
from MySageTokenizer import MySageTokenizer
from HFTokenizer import HFTokenizer
import csv


class Experiment:
    """
    Experiment object. Each experiment has 3 different tokenizers: l1, l2, l1_l2
    """
    def __init__(self, l1, l2, l1_training_corpus_dir, l2_training_corpus_dir, l1_words_dir, l2_words_dir, l1_l2_training_corpus_dir, algo_name, vocab_size, ff_words_dir, l1_tokenizer,
                 embedding_schedule=None, full_vocab_schedule=None):
        # Language 1, English
        self.l1 = l1
        # Language 2, Latin text
        self.l2 = l2
        self.l1_training_corpus_dir = l1_training_corpus_dir
        self.l2_training_corpus_dir = l2_training_corpus_dir
        # Words in corpus with their counts
        self.l1_words = self._read_corpus_words(l1_words_dir)
        self.l2_words = self._read_corpus_words(l2_words_dir)
        # Corpus that includes text from l1 and l2
        self.l1_l2_training_corpus_dir = l1_l2_training_corpus_dir
        self.algo_name = algo_name
        self.vocab_size = vocab_size
        # The False Friends words included in l1 and l2
        self.ff_data = self._read_ff_data(ff_words_dir)
        self.l1_tokenizer = l1_tokenizer
        # Used for SaGe tokenizer
        self.embedding_schedule = embedding_schedule
        # Used for SaGe tokenizer
        self.full_vocab_schedule = full_vocab_schedule
        self.l2_tokenizer = None
        self.l1_l2_tokenizer = None
        # The directory where the experiment object is saved
        self.main_dir = f"./experiments/{self.vocab_size}/{l2}"
        # Directory in which the experiment results will be saved
        self.analysis_dir = f"./analysis/{self.vocab_size}/{l2}"
        
    def __repr__(self):
        return f"{self.l2}_{self.algo_name}"
    
    def start_experiment(self):
        self._create_experiment_dir()
        self._train_tokenizers()
    
    def get_corpus_words(self, language):
        if language == self.l1:
            return self.l1_words
        elif language == self.l2:
            return self.l2_words
        else:
            print(f"Language {language} is not supported in experiment: {self.__repr__()}")
            
    def get_ff_words(self):
        ff_words = set()
        for i in range(len(self.ff_data)):
            ff_words.add(self.ff_data[i]["False Friend"])
        return ff_words
    
    def get_same_words_in_corpuses(self):
        l1_words = set(self.l1_words.keys())
        l2_words = set(self.l2_words.keys())
        return l1_words.intersection(l2_words)
        
    
    def get_tokenizers_list(self):
        return [self.l1_tokenizer, self.l2_tokenizer, self.l1_l2_tokenizer]
    
    def _read_corpus_words(self, path):
        """
        Get the word frequencies of words for language in the file path. Looks at all words as lower case, so the word
        "a" and "A" are considered the same
        :param path: word frequency file path
        :return: dictionary --> {word: word_frequency}
        """
        word_frequencies = dict()
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, freq = line.split("\t")[1:]
            # only lower case words
            word = word.lower()
            if word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] + int(freq.strip())
            else:
                word_frequencies[word] = int(freq.strip())
        return word_frequencies
    
    def _read_ff_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            # list of dictionaries
            return list(csv.DictReader(f))
       
    def _train_tokenizers(self):
        if "SAGE" in self.algo_name:
            self.l2_tokenizer = MySageTokenizer(self.l2, self.l2_training_corpus_dir, self.vocab_size, self.algo_name, self.embedding_schedule, self.full_vocab_schedule)
            self.l1_l2_tokenizer = MySageTokenizer(f"{self.l1}_{self.l2}", self.l1_l2_training_corpus_dir, self.vocab_size, self.algo_name,
                                                   self.embedding_schedule, self.full_vocab_schedule)
        else:
            self.l2_tokenizer = HFTokenizer(self.l2, self.l2_training_corpus_dir, self.vocab_size, self.algo_name)
            self.l1_l2_tokenizer = HFTokenizer(f"{self.l1}_{self.l2}", self.l1_l2_training_corpus_dir, self.vocab_size, self.algo_name)
        self.l2_tokenizer.train_tokenizer()
        self.l1_l2_tokenizer.train_tokenizer()
    
    def _create_experiment_dir(self):
        self._create_experiment_dir_helper(self.main_dir)
        self._create_experiment_dir_helper(self.analysis_dir)
        self._create_experiment_dir_helper(f"{self.analysis_dir}/graphs")
        self._create_experiment_dir_helper(f"{self.analysis_dir}/tokenization")
        if "SAGE" in self.algo_name:
            # self._create_experiment_dir_helper(f"./results/{self.l1}_{self.algo_name}_{self.vocab_size}")
            self._create_experiment_dir_helper(f"./results/{self.l2}_{self.algo_name}_{self.vocab_size}")
            self._create_experiment_dir_helper(f"./results/{self.l1}_{self.l2}_{self.algo_name}_{self.vocab_size}")

    def _create_experiment_dir_helper(self, path):
        os.makedirs(path, exist_ok=True)
        print(f"created directory {path}")
    
    def save_experiment(self):
        with open(f"{self.main_dir}/{self.__repr__()}.pkl", "wb") as f:
            pickle.dump(self, f)
    
    
    @classmethod
    def load_experiment(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    
    