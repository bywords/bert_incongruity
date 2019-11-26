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

def set_seed(args):
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output setups
    exp_id = "data-{}_model-{}_freeze-{}".format(args.data_dir, args.model, args.freeze)
    exp_dir = os.path.join(args.output_dir, exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, "logs.txt")
    model_path = os.path.join(exp_dir, args.model_file)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger("bert_incongruity")

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)

    logger.addHandler(f_handler)

    # Set seed
    set_seed(args)

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.max_epochs
    print(args.bert_type)
    tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)

    if args.model == "pool":
        model = BertPoolForIncongruity(args.bert_type, hidden_size=bert_dim(args.bert_type))
    elif args.model == "ahde":
        hidden_dims = \
            {'headline': args.headline_rnn_hidden_dim,
            'word': args.word_level_rnn_hidden_dim,
            'paragraph': args.word_level_rnn_hidden_dim
        }
        model = AttentionHDE(args.bert_type, hidden_dims, args.max_paragraph_num)
    else:
        raise ValueError("args.model should be set appropriately.")
    model.cuda()

    if args.freeze:
        model.freeze_bert_encoder()
    else:
        model.unfreeze_bert_encoder()

    # cannot shuffle with iterable dataset
    if args.model == "pool":
        test_set = \
            IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                       data_dir=args.data_dir, data_type=DataType.Test_sample)
        dev_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                             data_dir=args.data_dir, data_type=DataType.Dev_sample)

        training_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                  data_dir=args.data_dir, data_type=DataType.Train_sample)
    elif args.model =="ahde":
        test_set = \
            ParagraphIncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                data_dir=args.data_dir, data_type=DataType.Test,
                                                max_para_num=args.max_paragraph_num)
        dev_set = ParagraphIncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                      data_dir=args.data_dir, data_type=DataType.Dev,
                                                      max_para_num=args.max_paragraph_num)
        training_set = ParagraphIncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                                                           data_dir=args.data_dir, data_type=DataType.Train,
                                                           max_para_num=args.max_paragraph_num)
    else:
        raise ValueError("args.model should be set properly.")

    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)

    if args.mode == "train":

        # tokenizer, max_seq_len, filename
        train_dataloader = data.DataLoader(training_set, batch_size=args.batch_size)
        dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size)

        # Define optimizers
        optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.num_warmup_steps, t_total=args.num_total_steps)
        loss_fct = nn.BCEWithLogitsLoss()
        train_loss_set = []

        # trange is a tqdm wrapper around the normal python range
        for e_idx in range(1, epochs+1):

            logger.info("Epoch {} - Start Training".format(e_idx))

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0:
                    logger.info("Epoch {} - Iteration {}".format(e_idx, step*args.batch_size))
                # Add batch to GPU
                batch = tuplify_with_device(batch, device)
                # Unpack the inputs from our dataloader
                b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens,\
                b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens, \
                b_labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()

                # Forward pass
                b_logits = model(b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens,
                                 b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens)
                loss = loss_fct(b_logits.view(-1, 1), b_labels.view(-1, 1))
                train_loss_set.append(loss.item())

                # Backward pass
                loss.backward()

                # Update parameters and take a step using the computed gradient
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_head_input_ids.size(0)
                nb_tr_steps += 1

            logger.info("Epoch {} - Train loss: {:.4f}".format(e_idx, tr_loss / nb_tr_steps))

            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()
            logger.info("Epoch {} - Start Validation".format(e_idx))

            # Tracking variables
            dev_y_preds, dev_y_targets = [], []
            # Evaluate data for one epoch
            for batch in dev_dataloader:
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

                print(preds[0:5])
                print(label_ids[0:5])
                exit()

                # Move logits and labels to CPU
                preds = preds.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                dev_y_preds.append(preds)
                dev_y_targets.append(label_ids)

            dev_y_preds = np.concatenate(dev_y_preds).reshape((-1, ))
            dev_y_targets = np.concatenate(dev_y_targets).reshape((-1, )).astype(int)

            dev_acc = accuracy_score(dev_y_targets, dev_y_preds.round())
            dev_auroc = roc_auc_score(dev_y_targets, dev_y_preds)

            logger.info("Epoch {} - Dev Acc: {:.4f} AUROC: {:.4f}".format(e_idx, dev_acc, dev_auroc))

        torch.save(model.state_dict(), model_path)

    elif args.mode == "test":
        model = torch.load_state_dict(torch.load(model_path))

    else:
        logging.error("Wrong mode: {}".format(args.mode))
        raise TypeError("args.model should be train or test.")

    # Evaluate test data for one epoch
    y_targets, y_preds = [], []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuplify_with_device(batch, device)
        # Unpack the inputs from our dataloader
        b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens, \
        b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens, \
        b_num_paras, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            preds = torch.sigmoid(model(b_head_input_ids, b_head_token_type_ids, b_head_pool_masks, b_head_lens,
                                        b_body_input_ids, b_body_token_type_ids, b_body_pool_masks, b_body_lens))

        # Move logits and labels to CPU
        preds = preds.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        y_preds.append(preds)
        y_targets.append(label_ids)

    y_preds = np.concatenate(y_preds).reshape((-1, ))
    y_targets = np.concatenate(y_targets).reshape((-1, )).astype(int)

    acc = accuracy_score(y_targets, y_preds.round())
    auroc = roc_auc_score(y_targets, y_preds)
    logger.info("Test Accuracy: {:.4f}, AUROC: {:.4f}".format(acc, auroc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="mode: train / test")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="model: pool / ahde / nsp")
    parser.add_argument("--model_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")

    ## Other parameters
    parser.add_argument("--data_dir", default="data", type=str, help="root directory for output")
    parser.add_argument("--output_dir", default="output", type=str, help="root directory for output")
    parser.add_argument("--seed", default=1.0, type=float, help="floating value for random seed")
    parser.add_argument("--bert_type", default='bert-base-uncased', type=str,
                        help="bert pretrained model type. e.g., 'bert-base-uncased'")
    parser.add_argument("--freeze", default=False, type=bool, help="whether bert parameters are freezed")
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
