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

    def forward(self, headline_input_ids, bodytext_input_ids,
                headline_token_type_ids, bodytext_token_type_ids):
        headline_outputs = self.bert(headline_input_ids, token_type_ids=headline_token_type_ids)[0]  # last hidden states
        bodytext_outputs = self.bert(bodytext_input_ids, token_type_ids=bodytext_token_type_ids)[0]  # last hidden states

        headline_mean_hidden = headline_outputs.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = bodytext_outputs.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = torch.transpose(bodytext_mean_hidden, 1, 2)    # (batch, hidden_dim, 1)

        # (batch, 1)
        logits = torch.sigmoid(torch.matmul(torch.matmul(headline_mean_hidden, self.similarity),
                                            bodytext_mean_hidden) + self.similarity_bias)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True