# -*- encoding: utf-8 -*-
import os
import numpy as np
import enum
import torch
import pandas as pd
from io import StringIO
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, IterableDataset

bert_input_template = "[CLS] {} [SEP]"

# Using enum class create enumerations
class DataType(enum.Enum):
   Train = 1
   Dev = 2
   Test = 3
   Test_0 = 4
   Test_1 = 5
   Test_2 = 6
   Test_3 = 7


class IncongruityDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_type):
        'Initialization'
        if data_type == DataType.Train:
            path = os.path.join("data", "train.tsv")
        elif data_type == DataType.Dev:
            path = os.path.join("data", "dev.tsv")
        elif data_type == DataType.Test:
            path = os.path.join("data", "test.tsv")
        elif data_type == DataType.Test_0:
            path = os.path.join("data", "test_type_0.tsv")
        elif data_type == DataType.Test_1:
            path = os.path.join("data", "test_type_1.tsv")
        elif data_type == DataType.Test_2:
            path = os.path.join("data", "test_type_2.tsv")
        elif data_type == DataType.Test_3:
            path = os.path.join("data", "test_type_3.tsv")
        else:
            raise TypeError("data_type should be DataType class.")


        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["id", "headline", "body", "label", "fake_para_len", "fake_para_index", "fake_type"]

        # index, headline, body, label, length of sentence, [1, 2, ..., x]
        headlines, bodytexts, labels = [], [], df.label.values
        for idx, row in df.iterrows():
            headlines.append(tokenizer.tokenize(bert_input_template.format(row["headline"])))
            bodytexts.append(tokenizer.tokenize(bert_input_template.format(row["body"])))

        headlines = [tokenizer.convert_tokens_to_ids(x) for x in headlines]
        bodytexts = [tokenizer.convert_tokens_to_ids(x) for x in bodytexts]

        headlines = pad_sequences(headlines, maxlen=max_seq_len, dtype="long", truncating="post", padding="post")
        bodytexts = pad_sequences(bodytexts, maxlen=max_seq_len, dtype="long", truncating="post", padding="post")

        headlines = headlines[:, :max_seq_len]
        bodytexts = bodytexts[:, :max_seq_len]

        # Create attention masks
        head_attention_masks, body_attention_masks = [], []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in headlines:
            seq_mask = [float(i > 0) for i in seq]
            head_attention_masks.append(seq_mask)

        for seq in bodytexts:
            seq_mask = [float(i > 0) for i in seq]
            body_attention_masks.append(seq_mask)

        self.headline = headlines
        self.bodytext = bodytexts
        self.headline_mask = np.array(head_attention_masks)
        self.bodytext_mask = np.array(body_attention_masks)
        self.label = labels

        self.num_samples = len(df.index)

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        target_headline = self.headline[index, :]
        target_bodytext = self.bodytext[index, :]
        target_headline_mask = self.headline_mask[index, :]
        target_bodytext_mask = self.bodytext_mask[index, :]
        target_label = self.label[index, :]

        return target_headline, target_bodytext, target_headline_mask, target_bodytext_mask, target_label


class IncongruityIterableDataset(IterableDataset):
    'Characterizes an Iterabledataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_type):
        'Initialization'
        if data_type == DataType.Train:
            path = os.path.join("data", "train.tsv")
        elif data_type == DataType.Dev:
            path = os.path.join("data", "dev.tsv")
        elif data_type == DataType.Test:
            path = os.path.join("data", "test.tsv")
        elif data_type == DataType.Test_0:
            path = os.path.join("data", "test_type_0.tsv")
        elif data_type == DataType.Test_1:
            path = os.path.join("data", "test_type_1.tsv")
        elif data_type == DataType.Test_2:
            path = os.path.join("data", "test_type_2.tsv")
        elif data_type == DataType.Test_3:
            path = os.path.join("data", "test_type_3.tsv")
        else:
            raise TypeError("data_type should be DataType class.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):

        # Create an iterator
        file_itr = open(self.path)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def line_mapper(self, line):

        # Splits the line into text and label and applies preprocessing to the text
        df = pd.read_csv(StringIO(line), sep="\t", header=None)
        headline, bodytext, h_mask, b_mask, label = self.preprocess(df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3])

        return headline, bodytext, h_mask, b_mask, label

    def preprocess(self, headline, bodytext, label):
        headline = [self.tokenizer.convert_tokens_to_ids(x)
                    for x in self.tokenizer.tokenize(bert_input_template.format(headline))]
        bodytext = [self.tokenizer.convert_tokens_to_ids(x)
                    for x in self.tokenizer.tokenize(bert_input_template.format(bodytext))]
        headline = pad_sequences([headline], maxlen=self.max_seq_len,
                                 dtype="long", truncating="post", padding="post")[0, :]
        bodytext = pad_sequences([bodytext], maxlen=self.max_seq_len,
                                 dtype="long", truncating="post", padding="post")[0, :]
        headline_mask = np.array([float(i > 0) for i in headline])
        bodytext_mask = np.array([float(i > 0) for i in bodytext])

        return headline, bodytext, headline_mask, bodytext_mask, np.array([label])


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)