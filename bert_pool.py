# -*- encoding: utf-8 -*-
import torch
from transformers import BertModel


class BertForIncongruity(torch.nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config):
        super(BertForIncongruity, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.similarity = torch.randn(config.hidden_size, config.hidden_size)
        self.similarity_bias = torch.randn(1)

    def forward(self, headline_input_ids, bodytext_input_ids,
                headline_token_type_ids=None, bodytext_token_type_ids=None, labels=None):
        headline_outputs = self.bert(headline_input_ids, token_type_ids=headline_token_type_ids)
        bodytext_outputs = self.bert(bodytext_input_ids, token_type_ids=bodytext_token_type_ids)

        headline_mean_hidden = headline_outputs[0].mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = bodytext_outputs[0].mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        bodytext_mean_hidden = torch.transpose(bodytext_mean_hidden, 1, 2)    # (batch, hidden_dim, 1)

        bodytext_mean_hidden
        self.similarity()
        logits = self.get_logits(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True