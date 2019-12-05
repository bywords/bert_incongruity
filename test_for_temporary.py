# -*- encoding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
import logging

from torch import nn
from torch.utils import data
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from data_utils import ParagraphIncongruityIterableDataset, IncongruityIterableDataset, DataType, tuplify_with_device, bert_dim
from bert_pool import BertPoolForIncongruity
from bert_ahde import AttentionHDE


# To disable kears warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def set_seed(seed):
    n_gpu = torch.cuda.device_count()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output setups
    exp_id = "data-{}_model-pool_freeze-{}_seed-{}".format(args.data_dir, args.freeze, args.seed)
    exp_dir = os.path.join(args.output_dir, exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, "logs-each-type.txt")
    real_output_file = os.path.join(exp_dir, "BDE.txt")
    model_path = os.path.join(exp_dir, "model.pt")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("bert_incongruity")

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)

    logger.addHandler(f_handler)

    # Set seed
    set_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)

    model = BertPoolForIncongruity(args.bert_type, hidden_size=bert_dim(args.bert_type))
    model.cuda()

    if args.freeze:
        model.freeze_bert_encoder()
    else:
        model.unfreeze_bert_encoder()

    model.load_state_dict(torch.load(model_path))

    # for data_type in [DataType.Test_0, DataType.Test_1, DataType.Test_2, DataType.Test_3]:
    #     test_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
    #                                           data_dir=args.data_dir, data_type=data_type)
    #     test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)
    #
    #     # Evaluate test data for one epoch
    #     y_targets, y_preds = [], []
    #     for batch in test_dataloader:
    #         # Add batch to GPU
    #         batch = tuplify_with_device(batch, device)
    #         # Unpack the inputs from our dataloader
    #         b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens, \
    #         b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens, \
    #         b_labels = batch
    #         # Telling the model not to compute or store gradients, saving memory and speeding up validation
    #         with torch.no_grad():
    #             # Forward pass, calculate logit predictions
    #             preds = torch.sigmoid(model(b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens,
    #                                         b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens))
    #
    #         # Move logits and labels to CPU
    #         preds = preds.detach().cpu().numpy()
    #         label_ids = b_labels.to('cpu').numpy()
    #
    #         y_preds.append(preds)
    #         y_targets.append(label_ids)
    #
    #     y_preds = np.concatenate(y_preds).reshape((-1, ))
    #     y_targets = np.concatenate(y_targets).reshape((-1, )).astype(int)
    #
    #     temp_index = np.isnan(y_preds)
    #     y_preds = y_preds[~temp_index]
    #     y_targets = y_targets[~temp_index]
    #
    #     acc = accuracy_score(y_targets, y_preds.round())
    #     auroc = roc_auc_score(y_targets, y_preds)
    #     logger.info("Type: {}, Test Accuracy: {:.4f}, AUROC: {:.4f}".format(data_type, acc, auroc))


    for data_type in [DataType.Test_real]:
        test_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                              data_dir=args.data_dir, data_type=data_type)
        test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)

        # Evaluate test data for one epoch
        y_preds = []
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuplify_with_device(batch, device)
            # Unpack the inputs from our dataloader
            b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens, \
            b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens, \
            b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                preds = torch.sigmoid(model(b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens,
                                            b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens))

            # Move logits and labels to CPU
            preds = preds.detach().cpu().numpy()
            y_preds.append(preds)

        y_preds = np.concatenate(y_preds).reshape((-1,))

        temp_index = np.isnan(y_preds)
        y_preds = y_preds[~temp_index]

        with open(real_output_file, 'wt') as f:
            for idx, pred in enumerate(y_preds):
                print(idx, end=",", file=f)
                print(pred, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="data_new", type=str, help="root directory for data")
    parser.add_argument("--output_dir", default="output", type=str, help="root directory for output")
    parser.add_argument("--seed", default=1, type=int, help="integer value for random seed")
    parser.add_argument("--bert_type", default='bert-base-uncased', type=str,
                        help="bert pretrained model type. e.g., 'bert-base-uncased'")
    parser.add_argument("--freeze", default=True, type=bool, help="whether bert parameters are freezed")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max values for gradient clipping")
    parser.add_argument("--num_total_steps", default=1000, type=int, help="For AdamW Secheduler")
    parser.add_argument("--num_warmup_steps", default=100, type=int, help="For AdamW Secheduler")
    parser.add_argument("--max_seq_len", default=512, type=int, help="For AdamW Secheduler")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=4, type=int,
                        help="Number of max epochs for training.")
    parser.add_argument("--max_paragraph_num", default=30, type=int)
    parser.add_argument("--headline-rnn-hidden-dim", default=384, type=int)
    parser.add_argument("--word-level-rnn-hidden-dim", default=384, type=int)
    parser.add_argument("--paragraph-level-rnn-hidden-dim", default=384, type=int)
    parser.add_argument("--gpu_id", default=2, type=int, help="cuda device index")

    args = parser.parse_args()
    main(args)
