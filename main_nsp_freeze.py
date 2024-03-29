# -*- encoding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
import logging
import pandas as pd

from torch.utils import data
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import BertTokenizer, BertForNextSentencePrediction

from data_utils import NSP_RealworldDataset, NSP_IncongruityIterableDataset, DataType, \
    tuplify_with_device_for_nsp, tuplify_with_device_for_nsp_inference, str2bool


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
    exp_id = "bertNSP_data-{}_freezeTrue_seed-{}".format(args.data_dir.strip("/"), args.seed)

    exp_dir = os.path.join(args.output_dir, exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, "logs.txt")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("bertNSP")

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)

    logger.addHandler(f_handler)

    # Set seed
    set_seed(args.seed)

    # Number of training epochs (authors recommend between 2 and 4)
    tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)

    nsp_model = BertForNextSentencePrediction.from_pretrained(args.bert_type)
    nsp_model.cuda()

    # cannot shuffle with iterable dataset
    if args.mode == "test":
        test_set = NSP_IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                  data_dir=args.data_dir, data_type=DataType.Test)

    elif args.mode == "real_old":
        test_set = NSP_RealworldDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                        data_dir=args.data_dir, data_type=DataType.Test_real_old)

    elif args.mode == "real_new":
        test_set = NSP_RealworldDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                        data_dir=args.data_dir, data_type=DataType.Test_real_new)

    elif args.mode == "real_covid":
        test_set = NSP_RealworldDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                        data_dir=args.data_dir, data_type=DataType.Test_real_covid)

    else:
        raise TypeError("args.mode is wrong.")

    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)
    nsp_model.eval()

    if args.mode == "test":
        pred_output_path = os.path.join(exp_dir, "pred_test.tsv")

        # Evaluate test data for one epoch
        y_targets, y_preds = [], []
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuplify_with_device_for_nsp(batch, device)
            # Unpack the inputs from our dataloader
            b_tokens, b_attention_masks, b_segments_ids, b_labels = batch

            with torch.no_grad():
                prediction = nsp_model(b_tokens, attention_mask=b_attention_masks, token_type_ids=b_segments_ids)
                prediction = prediction[0]
                softmax = torch.nn.Softmax(dim=1)
                prediction_sm = softmax(prediction)

            preds = prediction_sm.detach().cpu().numpy()[:, 1]
            label_ids = b_labels.to('cpu').numpy().flatten()

            y_preds.append(preds)
            y_targets.append(label_ids)

        y_preds = np.concatenate(y_preds).reshape((-1, ))
        y_targets = np.concatenate(y_targets).reshape((-1, )).astype(int)

        temp_index = np.isnan(y_preds)
        y_preds = y_preds[~temp_index]
        y_targets = y_targets[~temp_index]

        acc = accuracy_score(y_targets, y_preds.round())
        auroc = roc_auc_score(y_targets, y_preds)
        logger.info("Test Accuracy: {:.4f}, AUROC: {:.4f}".format(acc, auroc))

        y_preds = pd.DataFrame({"pred": y_preds, "target": y_targets})
        y_preds.to_csv(pred_output_path, index=False, header=True, sep="\t")

    else: # for real world articles
        pred_output_path = os.path.join(exp_dir, "pred_{}.txt".format(args.mode))

        # Evaluate test data for one epoch
        y_targets, y_preds = [], []
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuplify_with_device_for_nsp_inference(batch, device)
            # Unpack the inputs from our dataloader
            b_tokens, b_attention_masks, b_segments_ids = batch

            with torch.no_grad():
                prediction = nsp_model(b_tokens, attention_mask=b_attention_masks, token_type_ids=b_segments_ids)
                prediction = prediction[0]
                softmax = torch.nn.Softmax(dim=1)
                prediction_sm = softmax(prediction)

            y_preds.append(prediction_sm.detach().cpu().numpy()[:, 1])

        y_preds = pd.Series(np.concatenate(y_preds).reshape((-1,)).tolist())
        y_preds.to_csv(pred_output_path, index=False, header=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="root directory for data")
    parser.add_argument("--mode", required=True, type=str,
                        help="mode: test / real_old / real_new / real_covid")

    ## Other parameters
    parser.add_argument("--output_dir", default="output", type=str, help="root directory for output")
    parser.add_argument("--seed", default=1, type=int, help="integer value for random seed")
    parser.add_argument("--bert_type", default='bert-base-uncased', type=str,
                        help="bert pretrained model type. e.g., 'bert-base-uncased'")
    parser.add_argument("--max_seq_len", default=512, type=int, help="maximum sequence lengths for BERT")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--max_paragraph_num", default=30, type=int)
    parser.add_argument("--gpu_id", default=2, type=int, help="cuda device index")
    parser.add_argument("--sample_train", default=False, type=str2bool, help="whether to sample a train file")

    args = parser.parse_args()
    main(args)
