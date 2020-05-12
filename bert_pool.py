# -*- encoding: utf-8 -*-
import torch
from torch import nn


class BertPoolForIncongruity(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, bert_model, hidden_size):
        super(BertPoolForIncongruity, self).__init__()
        self.bert = bert_model
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, headline_input_ids, headline_token_type_ids, headline_pool_masks, headline_lens,
                bodytext_input_ids, bodytext_token_type_ids, bodytext_pool_masks, bodytext_lens):
        headline_outputs = self.bert(headline_input_ids, attention_mask=headline_token_type_ids)[0]  # last hidden states
        bodytext_outputs = self.bert(bodytext_input_ids, attention_mask=bodytext_token_type_ids)[0]  # last hidden states

        headline_mean_hidden = \
            torch.div(torch.matmul(torch.transpose(headline_outputs, 1, 2), headline_pool_masks), headline_lens).squeeze()
        bodytext_mean_hidden = \
            torch.div(torch.matmul(torch.transpose(bodytext_outputs, 1, 2), bodytext_pool_masks), bodytext_lens).squeeze()

        # (batch, 1)
        return self.bilinear(headline_mean_hidden, bodytext_mean_hidden).view(-1, 1)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True