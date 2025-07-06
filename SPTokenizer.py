from MyTokenizer import MyTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle

class SPTokenizer(MyTokenizer):
    def __init__(self,language, training_corpus_dir, vocab_size, algo_name):
        self.language = language
        self.training_corpus_dir = training_corpus_dir
        self.vocab_size = vocab_size
        self.algo_name = algo_name
        self.tokenizer = None
        self.unk_token = "<UNK>"  # token for unknown words
        self.spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens
    
    def __repr__(self):
        return f"{self.language}_{self.vocab_size}_{self.algo_name}"
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens
    
    def train_tokenizer(self):
        if "BPE" in self.algo_name:
            tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
            trainer = BpeTrainer(special_tokens=self.spl_tokens, vocab_size=self.vocab_size)
        elif "UNI" in self.algo_name:
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(unk_token=self.unk_token, special_tokens=self.spl_tokens, vocab_size=self.vocab_size)
        elif "WPC" in self.algo_name:
            tokenizer = Tokenizer(WordPiece(unk_token=self.unk_token))
            trainer = WordPieceTrainer(special_tokens=self.spl_tokens, vocab_size=self.vocab_size)
        else:  # WLVL
            tokenizer = Tokenizer(WordLevel(unk_token=self.unk_token))
            trainer = WordLevelTrainer(special_tokens=self.spl_tokens, vocab_size=self.vocab_size)
        
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([self.training_corpus_dir], trainer)
        self.tokenizer = tokenizer
    
    def save_tokenizer(self, path):
        with open(f"{path}/{self.__repr__()}.pkl", "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_tokenizer(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_algo_name(self):
        return self.algo_name
    
    def get_training_corpus_dir(self):
        return self.training_corpus_dir
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_vocab(self):
        return self.tokenizer.get_vocab().keys()