from abc import ABC, abstractmethod

class MyTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass
    @abstractmethod
    def train_tokenizer(self):
        pass
    @abstractmethod
    def save_tokenizer(self, path):
        pass
    @classmethod
    @abstractmethod
    def load_tokenizer(self, path):
        pass
    @abstractmethod
    def get_algo_name(self):
        pass
    @abstractmethod
    def get_training_corpus_dir(self):
        pass
    @abstractmethod
    def get_vocab_size(self):
        pass
    @abstractmethod
    def get_vocab(self):
        pass