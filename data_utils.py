# -*- encoding: utf-8 -*-
import os, csv, sys
import numpy as np
import enum
import torch
import pandas as pd
from copy import deepcopy
from io import StringIO
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import IterableDataset, Dataset


bert_input_template = "[CLS] {} [SEP]"


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

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
   Test_real_old = 11
   Test_real_new = 12
   Test_real_covid = 13


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
        else:
            raise TypeError("data_type should be either train or test. Cannot be a type for real articles.")

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

        raw_headline = df.iloc[0, 1]
        raw_bodytext = df.iloc[0, 2]
        raw_label = df.iloc[0, 3]

        indexed_tokens, attention_masks, segment_ids, label = \
            self.preprocess(raw_headline, raw_bodytext, raw_label)

        return indexed_tokens, attention_masks, segment_ids, label

    def preprocess(self, headline, bodytext, label):
        indexed_tokens, attention_masks, segment_ids = pad_and_mask_for_bert_nsp(bodytext, headline,
                                                                                 self.tokenizer, self.max_seq_len)
        label = np.array(label).reshape(-1, 1)

        return indexed_tokens, attention_masks, segment_ids, label


class NSP_RealworldDataset(Dataset):
    'Characterizes an Iterabledataset for PyTorch'
    def __init__(self, tokenizer, max_seq_len, data_dir, data_type):
        'Initialization'
        if data_type == DataType.Test_real_old:
            path = os.path.join(data_dir, "real_world_old_no_EOS_EOP.tsv")
        elif data_type == DataType.Test_real_new:
            path = os.path.join(data_dir, "real_world_new_no_EOS_EOP.tsv")
        elif data_type == DataType.Test_real_covid:
            path = os.path.join(data_dir, "real_world_covid_no_EOS_EOP.tsv")
        else:
            raise TypeError("data_type should be one of the DataType classes for real world.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if data_type == DataType.Test_real_old:
            self.df = pd.read_csv(path, sep=",", header=None, encoding="utf-8")
        else:
            self.df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8", quoting=csv.QUOTE_NONE)

        self.headline = self.df.iloc[:, 1]
        self.bodytext = self.df.iloc[:, 2]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df.index)

    def __getitem__(self, index):

        raw_headline = self.headline[index]
        raw_bodytext = self.bodytext[index]

        indexed_tokens, attention_masks, segment_ids = \
            self.preprocess(raw_headline, raw_bodytext)

        return indexed_tokens, attention_masks, segment_ids

    def preprocess(self, headline, bodytext):
        indexed_tokens, attention_masks, segment_ids = pad_and_mask_for_bert_nsp(bodytext, headline,
                                                                                 self.tokenizer, self.max_seq_len)

        return indexed_tokens, attention_masks, segment_ids


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
        else:
            raise TypeError("data_type should be either train or test. Cannot be a type for real articles.")

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
            sys.exit()

        raw_headline = df.iloc[0, 1]
        raw_bodytext = df.iloc[0, 2]
        raw_label = df.iloc[0, 3]

        headline, h_mask, h_pool_mask, h_len, bodytext, b_mask, b_pool_mask, b_len, label = \
            self.preprocess(raw_headline, raw_bodytext, raw_label)

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


class RealworldDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len, data_dir, data_type):
        'Initialization'
        if data_type == DataType.Test_real_old:
            path = os.path.join(data_dir, "real_world_old_no_EOS_EOP.tsv")
        elif data_type == DataType.Test_real_new:
            path = os.path.join(data_dir, "real_world_new_no_EOS_EOP.tsv")
        elif data_type == DataType.Test_real_covid:
            path = os.path.join(data_dir, "real_world_covid_no_EOS_EOP.tsv")
        else:
            raise TypeError("data_type should be one of the DataType classes for real world.")

        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if data_type == DataType.Test_real_old:
            self.df = pd.read_csv(path, sep=",", header=None, encoding="utf-8")
        else:
            self.df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8", quoting=csv.QUOTE_NONE)

        self.headline = self.df.iloc[:, 1]
        self.bodytext = self.df.iloc[:, 2]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df.index)

    def __getitem__(self, index):

        raw_headline = self.headline[index]
        raw_bodytext = self.bodytext[index]

        headline, h_mask, h_pool_mask, h_len, bodytext, b_mask, b_pool_mask, b_len = \
            self.preprocess(raw_headline, raw_bodytext)

        return headline, h_mask, h_pool_mask, h_len, bodytext, b_mask, b_pool_mask, b_len

    def preprocess(self, headline, bodytext):
        headline, headline_mask, headline_pool_mask, headline_len = pad_and_mask_for_bert_emb(headline,
                                                                                              self.tokenizer,
                                                                                              self.max_seq_len)

        bodytext_parsed_str = " ".join(list(filter(lambda x: x not in ["<EOS>", "<EOP>"], bodytext.split())))
        bodytext, bodytext_mask, bodytext_pool_mask, bodytext_len = pad_and_mask_for_bert_emb(bodytext_parsed_str,
                                                                                              self.tokenizer,
                                                                                              self.max_seq_len)

        return headline, headline_mask, headline_pool_mask, headline_len, \
               bodytext, bodytext_mask, bodytext_pool_mask, bodytext_len


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

def tuplify_with_device_for_inference(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long), batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.float), batch[3].to(device, dtype=torch.float),
                  batch[4].to(device, dtype=torch.long), batch[5].to(device, dtype=torch.long),
                  batch[6].to(device, dtype=torch.float), batch[7].to(device, dtype=torch.float)])


def tuplify_with_device_for_nsp(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long),
                  batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.long),
                  batch[3].to(device, dtype=torch.float)])


def tuplify_with_device_for_nsp_inference(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long),
                  batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.long)])


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


def pad_and_mask_for_bert_nsp(text1, text2, tokenizer, max_seq_len):

    HEADLINE_MAX_LEN = 22
    BODYTEXT_MAX_LEN = 512 - 2 - HEADLINE_MAX_LEN

    text1_toks = ["[CLS]"] + tokenizer.tokenize(text1)[:BODYTEXT_MAX_LEN] + ["[SEP]"]
    text2_toks = tokenizer.tokenize(text2)[:HEADLINE_MAX_LEN]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(x) for x in text1_toks]
    indexed_tokens += [tokenizer.convert_tokens_to_ids(x) for x in text2_toks]

    indexed_tokens = pad_sequences([indexed_tokens], maxlen=max_seq_len,
                                   dtype="long", truncating="post", padding="post")[0, :]

    text1_len = len(text1_toks); text2_len = len(text2_toks); padded_len = max_seq_len - text1_len - text2_len
    segments_ids = [0] * text1_len + [1] * text2_len + [1] * padded_len
    segments_ids = np.array(segments_ids).reshape(-1, )

    attention_masks = [1] * (text1_len+text2_len) + [0] * padded_len
    attention_masks = np.array(attention_masks).reshape(-1, )

    return indexed_tokens, attention_masks, segments_ids
