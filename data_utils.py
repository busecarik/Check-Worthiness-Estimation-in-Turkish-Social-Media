import os
import re
import demoji
import torch
import numpy as np
import urlexpander
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

def expandURL(tweet_url):
    try:
        url = urlexpander.expand(tweet_url)
        parsed_url = re.findall('(\w+)://([\w\-\.]+)', url)[0]
        link = parsed_url[1]
        domain = link.split('.')[0]
        if domain == 'www':
            domain = link.split('.')[1]
        
        if domain == 'twitter':
            parsed_tweet = re.findall('([\w\-\.]+)/(\w+)', url)[0]
            return ' '.join(('$TWITTER', ''.join('$', str(parsed_tweet[1]))))
        else:
            return ''.join('$', domain.upper())
    except:
        return '$URL'

def preprocess(tweet, urlExpand=False):
    """remove links, emojis, usernames"""
    # Replace urls with URL token
    if urlExpand:
        tweet_urls = re.findall('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', tweet)
        for url in tweet_urls:
            tweets = t.replace(url, expandURL(url))
    else:
        tweet = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 'URL', tweet)

    # Replace usernames with USER token. The usernames are any word that starts with @.
    tweet = re.sub('\@[a-zA-Z0-9]*', '@USER', tweet)
    # Replacing/removing emojis
    tweet = demoji.replace(tweet, "")
    tweet = re.sub('(#[a-zA-Z0-9]*)', '', tweet) # remove hashtags
    return tweet

def _read_tsv(input_file, urlExpand=False):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        next(f)
        lines = []
        for line in f:
            line = line.split('\t')
            lines.append([preprocess(line[3]), int(line[4])])
        print("Loading data")
        print(len(lines))
    return np.array(lines) 

def convert_tokens_to_features(data, tokenizer, max_seq_len):
    labels = [int(example[1]) for example in data]
    examples = ["[CLS] " + example[0] + " [SEP]" for example in data]  
    tokenized_X = [tokenizer.tokenize(sent) for sent in examples]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_X]
    
    for i, input_id in enumerate(input_ids):
        if len(input_id) < max_seq_len:
            input_ids[i] += [0] * (max_seq_len - len(input_id))
        else: 
            print("The sentence is longer than the max_seq_length, it is truncating...")
            print(len(input_ids[i]))
            print(examples[i])
            input_ids[i] = input_ids[i][:max_seq_len]

    attention_masks = []
    for seq in input_ids:
        seq_mask = [int(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)

class BertDataModule(object):
    def __init__(self, 
        data_dir: str = 'data/', 
        model_name_or_path: str = "dbmdz/bert-base-turkish-cased", 
        max_seq_length: int = 150, 
        batch_size: int = 32,
        expandURL = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.expandURL = expandURL
        
    def load_datasets(self, urlExpand=False):
        self.train_dataset = convert_tokens_to_features(_read_tsv(os.path.join(self.data_dir, 'dataset_train_v1_turkish.tsv')), self.tokenizer, self.max_seq_length)
        self.val_dataset = convert_tokens_to_features(_read_tsv(os.path.join(self.data_dir, 'dataset_dev_v1_turkish.tsv')), self.tokenizer, self.max_seq_length)
        self.test_dataset = convert_tokens_to_features(_read_tsv(os.path.join(self.data_dir, 'dataset_test_v1_turkish.tsv')), self.tokenizer, self.max_seq_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)