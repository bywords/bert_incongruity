# -*- encoding: utf-8 -*-
import torch
from torch import nn
from transformers import BertModel


class BertPoolForIncongruity(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, vocab_file, hidden_size):
        super(BertPoolForIncongruity, self).__init__()
        self.bert = BertModel.from_pretrained(vocab_file)
        self.similarity = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.similarity_bias = nn.Parameter(torch.randn(1))

    def forward(self, headline_input_ids, headline_token_type_ids, headline_pool_masks, headline_lens,
                bodytext_input_ids, bodytext_token_type_ids, bodytext_pool_masks, bodytext_lens):
        headline_outputs = self.bert(headline_input_ids, token_type_ids=headline_token_type_ids)[0]  # last hidden states
        bodytext_outputs = self.bert(bodytext_input_ids, token_type_ids=bodytext_token_type_ids)[0]  # last hidden states

        print(headline_outputs.size())
        print(headline_pool_masks.size())
        print(headline_pool_masks.size())
        print(headline_lens.size())

        # print(headline_pool_masks[0])
        # print(headline_token_type_ids[0])

        temp = torch.matmul(torch.transpose(headline_outputs, 1, 2), headline_pool_masks).squeeze()
        print(temp.size())
        # headline_temp = headline_lens.expand(headline_lens.size(0), temp.size(1))
        # print(headline_temp.size())
        # temp2 = torch.div(temp, headline_temp)
        # print(temp2.size())
        temp3 = torch.div(temp, headline_lens)
        print(temp3.size())



        print(temp[0, 0:2])
        print(headline_lens[0])
        print(temp3[0,0:2])
        exit()


        # headline_outputs # (batch, seq, hidden_dim) --> (batch, hidden_dim, seq)
        # headline_pool_masks # (batch, seq, 1) --> (batch, seq, 1)
        #
        # # (batch, hidden_dim, 1) -- > (batch, hidden_dim)
        #
        # headline_lens # (batch, 1)

        headline_mean_hidden = headline_outputs.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = bodytext_outputs.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = torch.transpose(bodytext_mean_hidden, 1, 2)    # (batch, hidden_dim, 1)

        # (batch, 1)
        logits = torch.matmul(torch.matmul(headline_mean_hidden, self.similarity),
                              bodytext_mean_hidden) + self.similarity_bias

        return logits.view(-1, 1)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True