# -*- encoding: utf-8 -*-
import os, csv
import numpy as np
import enum
import torch
import pandas as pd
from copy import deepcopy
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
   Train_sample = 8
   Dev_sample = 9
   Test_sample = 10
   Test_real = 11


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
        elif data_type == DataType.Train_sample:
            path = os.path.join("data", "train_sample.tsv")
        elif data_type == DataType.Dev_sample:
            path = os.path.join("data", "dev_sample.tsv")
        elif data_type == DataType.Test_sample:
            path = os.path.join("data", "test_sample.tsv")
        elif data_type == DataType.Test_real:
            path = os.path.join("data", "real_world_articles_ascii.tsv")
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
        head_attention_masks, body_attention_masks, head_pooling_masks, body_pooling_masks = [], [], [], []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in headlines:
            seq_mask = [float(i > 0) for i in seq]
            pooling_mask = deepcopy(seq_mask)

            pooling_mask[pooling_mask.index(float(False))-1] = float(False)
            pooling_mask[0] = float(False)

            head_attention_masks.append(seq_mask)
            head_pooling_masks.append(pooling_mask)

        for seq in bodytexts:
            seq_mask = [float(i > 0) for i in seq]
            pooling_mask = deepcopy(seq_mask)

            pooling_mask[pooling_mask.index(float(False)) - 1] = float(False)
            pooling_mask[0] = float(False)

            body_attention_masks.append(seq_mask)
            body_pooling_masks.append(pooling_mask)

        self.headline = headlines
        self.bodytext = bodytexts
        self.headline_mask = np.array(head_attention_masks)
        self.bodytext_mask = np.array(body_attention_masks)
        self.headline_pool_mask = np.array(head_pooling_masks)
        self.bodytext_pool_mask = np.array(body_pooling_masks)
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
        target_headline_pooling_mask = self.headline_pool_mask[index, :]
        target_bodytext_mask = self.bodytext_mask[index, :]
        target_bodytext_pooling_mask = self.bodytext_pool_mask[index, :]
        target_label = self.label[index, :]

        return target_headline, target_bodytext, target_headline_mask, target_headline_pooling_mask,\
               target_bodytext_mask, target_bodytext_pooling_mask, target_label


class NSP_IncongruityIterableDataset(IterableDataset):
    'Characterizes an Iterabledataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_dir, data_type):
        'Initialization'
        if data_type == DataType.Train:
            path = os.path.join(data_dir, "train.tsv")
        elif data_type == DataType.Dev:
            path = os.path.join(data_dir, "dev.tsv")
        elif data_type == DataType.Test:
            path = os.path.join(data_dir, "test.tsv")
        elif data_type == DataType.Test_0:
            path = os.path.join(data_dir, "test_type_0.tsv")
        elif data_type == DataType.Test_1:
            path = os.path.join(data_dir, "test_type_1.tsv")
        elif data_type == DataType.Test_2:
            path = os.path.join(data_dir, "test_type_2.tsv")
        elif data_type == DataType.Test_3:
            path = os.path.join(data_dir, "test_type_3.tsv")
        elif data_type == DataType.Train_sample:
            path = os.path.join(data_dir, "train_sample.tsv")
        elif data_type == DataType.Dev_sample:
            path = os.path.join(data_dir, "dev_sample.tsv")
        elif data_type == DataType.Test_sample:
            path = os.path.join(data_dir, "test_sample.tsv")
        elif data_type == DataType.Test_real:
            path = os.path.join(data_dir, "real_world_articles_ascii.tsv")
        else:
            raise TypeError("data_type should be DataType class.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):

        # Create an iterator
        file_itr = open(self.path, encoding="utf-8")

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def line_mapper(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        try:
            df = pd.read_csv(StringIO(line), sep="\t", header=None, quoting=csv.QUOTE_NONE)
        except:
            print(line)
            exit()

        indexed_tokens, segment_masks, label = self.preprocess(df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3])

        return indexed_tokens, segment_masks, label

    def preprocess(self, headline, bodytext, label):
        indexed_tokens, segment_masks = pad_and_mask_for_bert_nsp(bodytext, headline, self.tokenizer)

        label = np.array(label).reshape(-1, 1)

        return indexed_tokens, segment_masks, label


class IncongruityIterableDataset(IterableDataset):
    'Characterizes an Iterabledataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_dir, data_type):
        'Initialization'
        if data_type == DataType.Train:
            path = os.path.join(data_dir, "train.tsv")
        elif data_type == DataType.Dev:
            path = os.path.join(data_dir, "dev.tsv")
        elif data_type == DataType.Test:
            path = os.path.join(data_dir, "test.tsv")
        elif data_type == DataType.Test_0:
            path = os.path.join(data_dir, "test_type_0.tsv")
        elif data_type == DataType.Test_1:
            path = os.path.join(data_dir, "test_type_1.tsv")
        elif data_type == DataType.Test_2:
            path = os.path.join(data_dir, "test_type_2.tsv")
        elif data_type == DataType.Test_3:
            path = os.path.join(data_dir, "test_type_3.tsv")
        elif data_type == DataType.Train_sample:
            path = os.path.join(data_dir, "train_sample.tsv")
        elif data_type == DataType.Dev_sample:
            path = os.path.join(data_dir, "dev_sample.tsv")
        elif data_type == DataType.Test_sample:
            path = os.path.join(data_dir, "test_sample.tsv")
        elif data_type == DataType.Test_real:
            path = os.path.join(data_dir, "real_world_articles_ascii.tsv")
        else:
            raise TypeError("data_type should be DataType class.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):

        # Create an iterator
        file_itr = open(self.path, encoding="utf-8")

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def line_mapper(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        try:
            df = pd.read_csv(StringIO(line), sep="\t", header=None, quoting=csv.QUOTE_NONE)
        except:
            print(line)
            exit()

        headline, h_mask, h_pool_mask, h_len, bodytext, b_mask, b_pool_mask, b_len, label = \
            self.preprocess(df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3])

        return headline, h_mask, h_pool_mask, h_len, bodytext, b_mask, b_pool_mask, b_len, label

    def preprocess(self, headline, bodytext, label):
        headline, headline_mask, headline_pool_mask, headline_len = pad_and_mask_for_bert_emb(headline,
                                                                                              self.tokenizer,
                                                                                              self.max_seq_len)

        bodytext_parsed_str = " ".join(list(filter(lambda x: x not in ["<EOS>", "<EOP>"], bodytext.split())))
        bodytext, bodytext_mask, bodytext_pool_mask, bodytext_len = pad_and_mask_for_bert_emb(bodytext_parsed_str,
                                                                                              self.tokenizer,
                                                                                              self.max_seq_len)

        label = np.array(label).reshape(-1, 1)

        return headline, headline_mask, headline_pool_mask, headline_len, \
               bodytext, bodytext_mask, bodytext_pool_mask, bodytext_len, \
               label


class ParagraphIncongruityIterableDataset(IterableDataset):
    'Characterizes an Iterabledataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_dir, data_type, max_para_num):
        'Initialization'
        if data_type == DataType.Train:
            path = os.path.join(data_dir, "train.tsv")
        elif data_type == DataType.Dev:
            path = os.path.join(data_dir, "dev.tsv")
        elif data_type == DataType.Test:
            path = os.path.join(data_dir, "test.tsv")
        elif data_type == DataType.Test_0:
            path = os.path.join(data_dir, "test_type_0.tsv")
        elif data_type == DataType.Test_1:
            path = os.path.join(data_dir, "test_type_1.tsv")
        elif data_type == DataType.Test_2:
            path = os.path.join(data_dir, "test_type_2.tsv")
        elif data_type == DataType.Test_3:
            path = os.path.join(data_dir, "test_type_3.tsv")
        elif data_type == DataType.Train_sample:
            path = os.path.join(data_dir, "train_sample.tsv")
        elif data_type == DataType.Dev_sample:
            path = os.path.join(data_dir, "dev_sample.tsv")
        elif data_type == DataType.Test_sample:
            path = os.path.join(data_dir, "test_sample.tsv")
        else:
            raise TypeError("data_type should be DataType class.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_para_num = max_para_num

    def __iter__(self):

        # Create an iterator
        file_itr = open(self.path)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def line_mapper(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        df = pd.read_csv(StringIO(line), sep="\t", header=None)
        headline, h_mask, h_pool_mask, h_len, paragraphs, p_mask, p_pool_mask, p_len, label = \
            self.preprocess(df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3])

        return headline, h_mask, h_pool_mask, h_len, paragraphs, p_mask, p_pool_mask, p_len, label

    def preprocess(self, headline, bodytext, label):
        headline, headline_mask, headline_pool_mask, headline_len = pad_and_mask_for_bert_emb(headline,
                                                                                              self.tokenizer,
                                                                                              self.max_seq_len)

        bodytext_parsed_str = " ".join(list(filter(lambda x: x not in ["<EOS>"], bodytext.split())))
        paragraphs, paragraphs_mask, paragraphs_pool_mask, paragraphs_len = [], [], [], []

        for paragraph in bodytext_parsed_str.split("<EOP>"):
            # split body texts into multiple paragraphs
            text, mask, pool_mask, text_len = \
                pad_and_mask_for_bert_emb(paragraph.strip(), self.tokenizer, self.max_seq_len)

            paragraphs.append(text)
            paragraphs_mask.append(mask)
            paragraphs_pool_mask.append(pool_mask)
            paragraphs_len.append(text_len)

        num_paragraphs = len(paragraphs_len)
        null_text = np.zeros_like(text)
        null_mask = np.zeros_like(mask)
        null_pool_mask = np.zeros_like(pool_mask)
        null_text_len = np.zeros_like(text_len)

        if self.max_para_num > num_paragraphs:
            for _ in range(self.max_para_num - num_paragraphs):
                paragraphs.append(null_text)
                paragraphs_mask.append(null_mask)
                paragraphs_pool_mask.append(null_pool_mask)
                paragraphs_len.append(null_text_len)

        paragraphs = np.array(paragraphs)
        paragraphs_mask = np.array(paragraphs_mask)
        paragraphs_pool_mask = np.array(paragraphs_pool_mask)
        paragraphs_len = np.array(paragraphs_len)

        paragraphs = paragraphs[:self.max_para_num]
        paragraphs_mask = paragraphs_mask[:self.max_para_num]
        paragraphs_pool_mask = paragraphs_pool_mask[:self.max_para_num]
        paragraphs_len = paragraphs_len[:self.max_para_num]

        label = np.array(label).reshape(-1, 1)

        return headline, headline_mask, headline_pool_mask, headline_len, \
               paragraphs, paragraphs_mask, paragraphs_pool_mask, paragraphs_len, \
               label


def tuplify_with_device(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long), batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.float), batch[3].to(device, dtype=torch.float),
                  batch[4].to(device, dtype=torch.long), batch[5].to(device, dtype=torch.long),
                  batch[6].to(device, dtype=torch.float), batch[7].to(device, dtype=torch.float),
                  batch[8].to(device, dtype=torch.float)])


def tuplify_with_device_for_nsp(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long),
                  batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.float)])


def bert_dim(bert_model):
    if bert_model == "bert-base-uncased":
        dim = 768
    else:
        raise ValueError("bert_model should be one of the pre-specificed settings.")

    return dim


def pad_and_mask_for_bert_emb(text, tokenizer, max_seq_len):
    text = [tokenizer.convert_tokens_to_ids(x) for x in tokenizer.tokenize(bert_input_template.format(text))]
    text = pad_sequences([text], maxlen=max_seq_len, dtype="long", truncating="post", padding="post")[0, :]

    token_out_of_seq = float(False)
    mask = [float(i > 0) for i in text]
    pool_mask = deepcopy(mask)

    if token_out_of_seq in pool_mask:
        pool_mask[pool_mask.index(token_out_of_seq) - 1] = token_out_of_seq

    pool_mask[0] = token_out_of_seq
    pool_mask = np.array(pool_mask)
    pool_mask = np.array(pool_mask).reshape(-1, 1)
    text_len = np.array(pool_mask.sum()).reshape(-1, 1)
    mask = np.array(mask)

    return text, mask, pool_mask, text_len


def pad_and_mask_for_bert_nsp(text1, text2, tokenizer):

    text1_toks = ["[CLS]"] + tokenizer.tokenize(text1)[:450] + ["[SEP]"]
    text2_toks = tokenizer.tokenize(text2)[:60]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in text1_toks + text2_toks]
    indexed_tokens = \
        pad_sequences([indexed_tokens], maxlen=len(indexed_tokens),
                      dtype="long", truncating="post", padding="post")[0, :]

    segments_ids = [0] * len(text1_toks) + [1] * len(text2_toks)
    segments_ids = np.array(segments_ids).reshape(-1, 1)

    return indexed_tokens, segments_ids
