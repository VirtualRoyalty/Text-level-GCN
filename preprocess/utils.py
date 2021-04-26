import os
import re
import numpy as np

from tqdm import tqdm
from nltk.tokenize import word_tokenize


def build_vocab(corpus, min_tf=5, max_tf=1000):
    term2freq = {}

    for doc in tqdm(range(len(corpus))):
      tokens = word_tokenize(doc)
      for token in tokens:
        if token in term2freq:
          term2freq[token] += 1
        else:
          term2freq[token] = 1

    vocab_terms = []
    for term in tqdm(term2freq):
      if MIN_TF < term2freq[term] < MAX_TF:
        vocab_terms.append(term)

    return vocab_terms
