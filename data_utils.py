import os
import torch
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()
        self.total = 0

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        token_id = self.char2idx[char]
        self.counter[token_id] += 1
        self.total += 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.data = self.tokenize(path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add chars to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                chars = list(line)
                tokens += len(chars)
                for char in chars:
                    self.dictionary.add_char(char)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                chars = list(line)
                for char in chars:
                    ids[token] = self.dictionary.char2idx[char]
                    token += 1

        return ids
