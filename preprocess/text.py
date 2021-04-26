import os
import re
import numpy as np

from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from nltk.tokenize import word_tokenize


class StringProcessor(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        # lemmatizer = WordNetLemmatizer()
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            import spacy
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token =  token.lemma_ # lemmatizer.lemmatize(token.text) #

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = " ".join(re.split(' +|\n+', result)).strip()
        return result

    def remove_short(self, string, min_len=2):
        return " ".join([word for word in string.split() if len(word)>min_len])


class CorpusProcessor:
    def __init__(self,
                 df,
                 save_path,
                 pipeline=['clean',
                           'remove_stopword',
                           'normalize',
                           'remove_short'],
                 encoding=None):
        self.save_path = save_path
        self.df = df
        self.pipeline = pipeline
        self.encoding = encoding
        self.processor = StringProcessor()

    def run(self):
        self.clean_text()
        self.save()

    def clean_text(self):

        clean_text_lst = []
        for indx in tqdm(range(len(self.df)), desc="processing", position=0):
            item = self.df['text'].iloc[indx]
            data = item.strip()
            if 'clean' in self.pipeline:
                data = self.processor.clean_str(data)
            if 'remove_stopword' in self.pipeline:
                data = self.processor.remove_stopword(data)
            if 'normalize' in  self.pipeline:
                data =  self.processor.norm_str(data)
            if 'remove_short' in  self.pipeline:
                data =  self.processor.remove_short(data)
            clean_text_lst.append(data)
        self.df['clean_text'] = clean_text_lst
        return self.df

    def save(self):
        self.df.to_csv(self.save_path)
        return

    from tqdm.notebook import tqdm
