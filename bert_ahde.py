# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from transformers import BertModel

from data_utils import bert_dim


class ParaHeadlineAttention(nn.Module):
    def __init__(self, para_dim, headline_dim, hidden_dim):
        super().__init__()
        self.linear_para = nn.Linear(para_dim, hidden_dim, bias=False)
        self.linear_headline = nn.Linear(headline_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_dim))  # [H_hidden]

    def forward(self, paras, para_mask, headline):
        # paras: [N, P, H_para]
        # para_mask: [N, P]
        # headline: [N, H_headline]
        z = self.linear_para(paras) + self.linear_headline(headline).unsqueeze(1)  # [N, P, H_hidden]
        s = torch.tanh(z).matmul(self.v)  # [N, P]
        s[para_mask] = float('-inf')
        a = torch.softmax(s, dim=1)  # [N, P]
        return torch.matmul(a.unsqueeze(1), paras).squeeze()  # paras: [N, H_para]


class AttentionHDE(nn.Module):
    def __init__(self, bert_model, hidden_dims, max_para_num):
        super(AttentionHDE, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.embedding_dim = bert_dim(bert_model)

        self.head_transform = nn.Linear(self.embedding_dim, hidden_dims["headline"])
        self.body_transform = nn.Linear(self.embedding_dim, hidden_dims["word"])

        self.body_encoder = nn.GRU(hidden_dims['word'], hidden_dims['paragraph'], bidirectional=True)
        self.attention = ParaHeadlineAttention(2 * hidden_dims['paragraph'], hidden_dims['headline'],
                                               2 * hidden_dims['paragraph'])
        self.bilinear = nn.Bilinear(hidden_dims['headline'], 2 * hidden_dims['paragraph'], 1)

        self.max_para_num = max_para_num

    #def forward(self, headlines, headline_lengths, bodys, para_lengths):
    def forward(self, headline_input_ids, headline_token_type_ids, headline_pool_masks, headline_lens,
                bodytext_input_ids, bodytext_token_type_ids, bodytext_pool_masks, bodytext_lens,
                para_lengths):
        #  headline_input_ids, headline_token_type_ids, headline_pool_masks, headline_lens,
        #                 bodytext_input_ids, bodytext_token_type_ids, bodytext_pool_masks, bodytext_lens


        # headline: [N, L_headline], 
        # headline_lengths: [N],
        # bodys: [N, P, L_para], 
        # para_lengths: [N, P]
        #
        # Note
        #  - GRU inputs: input [seq_len, batch, input_size], h_0 [num_layers * num_directions, batch, hidden_size]
        #  - GRU outputs: output [seq_len, batch, num_directions * hidden_size], h_n [num_layers * num_directions, batch, hidden_size]

        headline_outputs = self.bert(headline_input_ids, token_type_ids=headline_token_type_ids)[0]
        headline_mean_hidden = \
            torch.div(torch.matmul(torch.transpose(headline_outputs, 1, 2), headline_pool_masks), headline_lens)
        x_headline = headline_mean_hidden.transpose(1, 2)  # (batch,
        x_headline = self.head_transform(x_headline)

        print(x_headline.shape)

        # achieve the hidden vector for every paragraph of body text
        bodytext_input_ids_chunks = torch.chunk(bodytext_input_ids, chunks=self.max_para_num, dim=1)
        bodytext_token_type_ids_chunks = torch.chunk(bodytext_token_type_ids, chunks=self.max_para_num, dim=1)
        bodytext_pool_masks_chunks = torch.chunk(bodytext_pool_masks, chunks=self.max_para_num, dim=1)
        bodytext_lens_chunks = torch.chunk(bodytext_lens, chunks=self.max_para_num, dim=1)

        x_bodytext_chunks = []
        for bodytext_input_id, bodytext_token_type_id, bodytext_pool_mask, bodytext_len in \
                zip(bodytext_input_ids_chunks, bodytext_token_type_ids_chunks, bodytext_pool_masks_chunks, bodytext_lens_chunks):

            bodytext_input_id = bodytext_input_id.squeeze(dim=1)
            bodytext_token_type_id = bodytext_token_type_id.squeeze(dim=1)
            bodytext_pool_mask = bodytext_pool_mask.squeeze(dim=1)
            bodytext_len = bodytext_len.squeeze(dim=1)

            bodytext_outputs = self.bert(bodytext_input_id, token_type_ids=bodytext_token_type_id)[0]
            bodytext_mean_hidden = \
                torch.div(torch.matmul(torch.transpose(bodytext_outputs, 1, 2), bodytext_pool_mask), bodytext_len)
            x_bodytext = bodytext_mean_hidden.transpose(1, 2)

            x_bodytext_chunks.append(x_bodytext.unsqueeze(dim=1))

        x_bodytext = torch.cat(x_bodytext_chunks, dim=1)
        print(x_bodytext.shape)
        exit()





        x_body = self.word_embeds(bodys)  # [N, P, L_para, H_embed]

        _, h_headline = self.headline_encoder(pack_padded_sequence(x_headline, headline_lengths,
                                                                   batch_first=True, enforce_sorted=False))  # _, [1, N, H_enc]
        h_headline = h_headline.squeeze()  # [N, H_enc]

        valid_para_lengths = (para_lengths != 0).sum(dim=1).tolist()
        para_mask = (para_lengths == 0)
        # merge dimensions N and P
        para_lengths = para_lengths.flatten()
        para_lengths_masked = para_lengths[para_lengths != 0]
        x_paras_masked = x_body.flatten(0,1)[para_lengths != 0]

        _, h_paras_masked = self.paragraph_encoder(pack_padded_sequence(x_paras_masked, para_lengths_masked, 
                                                                        batch_first=True, enforce_sorted=False))

        # unmerge dimensions N and P
        h_paras_grouped = h_paras_masked.squeeze().split(valid_para_lengths)
        h_paras = pad_sequence(h_paras_grouped)

        output_body_packed, _ = self.body_encoder(pack_padded_sequence(h_paras, valid_para_lengths, enforce_sorted=False))
        output_body, _ = pad_packed_sequence(output_body_packed, 
                                             batch_first=True, total_length=para_mask.shape[-1])  # [N, P, 2 * H]

        h_body = self.attention(output_body, para_mask, h_headline)

        return self.bilinear(h_headline, h_body).squeeze()

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
