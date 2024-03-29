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
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForNextSentencePrediction

from data_utils import NSP_IncongruityIterableDataset, DataType, tuplify_with_device_for_nsp


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
    exp_id = "bertNSP_data-{}_freeze-False_seed-{}".format(args.data_dir.strip("/"),
                                                           args.seed)

    exp_dir = os.path.join(args.output_dir, exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, "logs.txt")
    model_path = os.path.join(exp_dir, args.model_file)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(exp_id)

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)

    logger.addHandler(f_handler)

    # Set seed
    set_seed(args.seed)

    if "uncased" in args.bert_type:
        tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=False)

    # cannot shuffle with iterable dataset
    test_set = NSP_IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                              data_dir=args.data_dir, data_type=DataType.Test)
    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)

    if args.mode == "train":
        epochs = args.max_epochs
        nsp_model = BertForNextSentencePrediction.from_pretrained(args.bert_type)
        nsp_model.cuda()

        training_set = NSP_IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                      data_dir=args.data_dir, data_type=DataType.Train)
        train_dataloader = data.DataLoader(training_set, batch_size=args.batch_size)
        dev_set = NSP_IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                 data_dir=args.data_dir, data_type=DataType.Dev)
        dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size)

        # Define optimizers
        optimizer = AdamW(nsp_model.parameters(), lr=args.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                    num_training_steps=args.num_total_steps)
        train_loss_set = []
        for e_idx in range(1, epochs + 1):

            logger.info("Epoch {} - Start Training".format(e_idx))

            # Set our model to training mode (as opposed to evaluation mode)
            nsp_model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0:
                    logger.info("Epoch {} - Iteration {}".format(e_idx, step * args.batch_size))
                # Add batch to GPU
                batch = tuplify_with_device_for_nsp(batch, device)
                # Unpack the inputs from our dataloader
                b_indexed_tokens, b_attention_masks, b_segments_ids, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()

                print(b_indexed_tokens.size())
                print(b_attention_masks.size())
                print(b_segments_ids.size())
                print(b_labels.size())

                # Forward pass
                loss, _ = nsp_model(b_indexed_tokens, attention_mask=b_attention_masks,
                                    token_type_ids=b_segments_ids, next_sentence_label=b_labels)
                train_loss_set.append(loss.item())

                # Backward pass
                loss.backward()

                # Update parameters and take a step using the computed gradient
                nn.utils.clip_grad_norm_(nsp_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_indexed_tokens.size(0)
                nb_tr_steps += 1

            logger.info("Epoch {} - Train loss: {:.4f}".format(e_idx, tr_loss / nb_tr_steps))

            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            nsp_model.eval()
            logger.info("Epoch {} - Start Validation".format(e_idx))

            # Tracking variables
            dev_y_preds, dev_y_targets = [], []
            # Evaluate data for one epoch
            for batch in dev_dataloader:
                # Add batch to GPU
                batch = tuplify_with_device_for_nsp(batch, device)
                # Unpack the inputs from our dataloader
                b_indexed_tokens, b_attention_masks, b_segment_ids, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    loss, predictions = nsp_model(b_indexed_tokens, attention_mask=b_attention_masks,
                                                  token_type_ids=b_segment_ids, next_sentence_label=b_labels)
                    softmax = torch.nn.Softmax(dim=1)
                    predictions_sm = softmax(predictions)

                # Move logits and labels to CPU
                preds = predictions_sm.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                dev_y_preds.append(preds)
                dev_y_targets.append(label_ids)

            dev_y_preds = np.concatenate(dev_y_preds).reshape((-1,))
            dev_y_targets = np.concatenate(dev_y_targets).reshape((-1,)).astype(int)

            dev_acc = accuracy_score(dev_y_targets, dev_y_preds.round())
            dev_auroc = roc_auc_score(dev_y_targets, dev_y_preds)

            logger.info("Epoch {} - Dev Acc: {:.4f} AUROC: {:.4f}".format(e_idx, dev_acc, dev_auroc))

        torch.save(nsp_model.state_dict(), model_path)

    elif args.mode == "test":
        nsp_model = BertForNextSentencePrediction.from_pretrained(model_path)
        nsp_model.cuda()

    else:
        logging.error("Wrong mode: {}".format(args.mode))
        raise TypeError("args.model should be train or test.")

    nsp_model.eval()

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

        preds = prediction_sm.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="root directory for data")
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="mode: train / test")
    parser.add_argument("--model_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")

    ## Other parameters
    parser.add_argument("--output_dir", default="output", type=str, help="root directory for output")
    parser.add_argument("--seed", default=1, type=int, help="integer value for random seed")
    parser.add_argument("--bert_type", default='bert-base-uncased', type=str,
                        help="bert pretrained model type. e.g., 'bert-base-uncased'")
    parser.add_argument("--gpu_id", default=2, type=int, help="cuda device index")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max values for gradient clipping")
    parser.add_argument("--num_total_steps", default=1000, type=int, help="For AdamW Secheduler")
    parser.add_argument("--num_warmup_steps", default=100, type=int, help="For AdamW Secheduler")
    parser.add_argument("--max_seq_len", default=512, type=int, help="maximum sequence lengths")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=2, type=int,
                        help="Number of max epochs for training.")

    args = parser.parse_args()
    main(args)
