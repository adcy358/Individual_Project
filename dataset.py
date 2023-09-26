import torch 
from torch import nn 
import pandas as pd
from collections import Counter
from torchtext.data.utils import get_tokenizer
import spacy

class SpanishPoemsDataset(torch.utils.data.Dataset): 
    
    def __init__(self, dir_path, start_token, end_token, args): 
        self.args = args
        self.start_token = start_token
        self.end_token = end_token
        self.dir_path = dir_path
        
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()
        
        # {index: word}
        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        # {word: index}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]
        
    def tokens_to_text(self, st_text, stop_words, use_stop_words): 
        l=[]
        for s in st_text: # loop through each sentence
            s = [x for x in s if x.isalnum()]
            s.insert(0, self.start_token) # insert start/end tokens
            s.append(self.end_token)
            l.extend(s)  
        return l
    
    def load_words(self):
        
        data = pd.read_csv(self.dir_path)
        #poems = data.content
        poems = data.content[:200]
        sent_corpus = [line.lower() for poem in poems for line in str(poem).split("\n")]
        
        tokenizer = get_tokenizer('spacy', language='es_core_news_sm')
        list_sent_corpus = [tokenizer(x) for x in sent_corpus if x != ''] # remove the spaces
        
        word_corpus = self.tokens_to_text(list_sent_corpus, self.start_token, self.end_token)
     
        return word_corpus
    
    def get_unique_words(self): 
        word_count = Counter(self.words)
        return sorted(word_count, key=word_count.get, reverse=True)
        
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
    
    def __getitem__(self, index): 
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )