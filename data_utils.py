# -*- encoding: utf-8 -*-
import os
import numpy as np
import enum
import torch
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from torch.utils import data

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


class IncongruityDataset(data.Dataset):
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

        print(headlines.shape)
        print(bodytexts.shape)
        exit()

        # Create attention masks
        head_attention_masks, body_attention_masks = [], []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in headlines:
            seq_mask = [float(i > 0) for i in seq]
            head_attention_masks.append(seq_mask)

        for seq in bodytexts:
            seq_mask = [float(i > 0) for i in seq]
            body_attention_masks.append(seq_mask)

        #self.headlines =

        self.df = pd.DataFrame({"headline": headlines, "bodytext": bodytexts,
                                "headline_mask": head_attention_masks, "bodytext_mask": body_attention_masks,
                                "label": labels})

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df.index)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        target_df = self.df.iloc[index, :]

        return target_df.headline.values, target_df.bodytext.values, \
               target_df.headline_mask.values, target_df.headline_mask.values, \
               target_df.label.values


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)