import os
import re
import numpy as np

from tqdm import tqdm, tqdm_notebook
from nltk.tokenize import word_tokenize


def build_vocab(corpus, min_tf=5, max_tf=1000, pad_token='PAD', master_node=False, master_token='MASTER_NODE'):
    term2freq = {}

    for tokens in tqdm_notebook(corpus):
      # tokens = word_tokenize(doc)
      for token in tokens:
        if token in term2freq:
          term2freq[token] += 1
        else:
          term2freq[token] = 1

    vocab_terms = []

    term2id = {}
    term_ind = 0
    for term in term2freq:
      if min_tf < term2freq[term] < max_tf:
        vocab_terms.append(term)
        term2id[term] = term_ind
        term_ind += 1

    term2id[pad_token] = len(vocab_terms)
    vocab_terms.append(pad_token)


    if master_node:
        term2id[master_token] = len(vocab_terms)
        vocab_terms.append(master_token)


    return vocab_terms, term2id

def get_max_nodes(df, token_col='tokens', quantile=0.95, master_node=False):
    df['n_unique'] = df[token_col].apply(lambda x: len(set(x)))
    max_nodes = int(df['n_unique'].quantile(quantile))
    if master_node:
      max_nodes += 1
    return max_nodes


import tensorflow.keras as keras
from tensorflow.keras import layers, Input
from IPython.display import clear_output
import matplotlib.pyplot as plt

class PlotLosses(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.losses = []
    self.val_losses = []
    self.accuracy = []
    self.val_accuracy = []
    self.logs = []

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    self.losses.append(logs.get('loss'))
    self.accuracy.append(logs.get('accuracy'))
    self.val_losses.append(logs.get('val_loss'))
    self.val_accuracy.append(logs.get('val_accuracy'))
    self.i += 1

    clear_output(wait=True)
    # plt.figure(figsize=(10, 5))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(self.x, self.losses, label="lr="+str(logs.get("lr"))+"\nloss")
    axes[0].plot(self.x, self.val_losses, label="val_loss")
    axes[1].plot(self.x, self.accuracy, '--',label="accuracy")
    axes[1].plot(self.x, self.val_accuracy, '--', label="val_accuracy")
    axes[0].grid()
    axes[1].grid()
    axes[0].set(xlabel='epoch', ylabel='loss')
    axes[0].set_title('Loss')
    axes[0].legend(shadow=True, fancybox=True, title='...')
    axes[1].set(xlabel='epoch', ylabel='accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(shadow=True, fancybox=True, title='...')
    plt.show()
