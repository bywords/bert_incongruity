# -*- encoding: utf-8 -*-
import numpy as np
import random
import argparse
import torch
import logging
from torch import nn, data
from tqdm import trange
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from data_utils import IncongruityDataset, DataType, flat_accuracy
from bert_pool import BertPoolForIncongruity


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)


    # Parameters:
    lr = 1e-3
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # tokenizer, max_seq_len, filename
    training_set = IncongruityDataset(tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN, data_type=DataType.Train)
    train_dataloader = data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

    dev_set = IncongruityDataset(tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN, data_type=DataType.Dev)
    dev_dataloader = data.DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False)

    test_set = IncongruityDataset(tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN, data_type=DataType.Dev)
    test_dataloader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = BertPoolForIncongruity('bert-base-uncased', do_lower_case=True, hidden_dim=HIDDEN_DIM)
    model.cuda()

    if args.freeze:
        model.freeze_bert_encoder()
    else:
        model.unfreeze_bert_encoder()

    # Define optimizers
    optimizer = AdamW(model.parameters(), lr=lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)
    loss_fct = nn.BCEWithLogitsLoss()
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_head_input_ids, b_body_input_ids, b_head_token_type_ids, b_body_token_type_ids, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass
            logits = model(b_head_input_ids, b_body_input_ids, b_head_token_type_ids, b_body_token_type_ids)
            loss = loss_fct(logits.view(-1, 1), b_labels.view(-1, 1))
            train_loss_set.append(loss.item())

            # Backward pass
            loss.backward()

            # Update parameters and take a step using the computed gradient
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scheduler.step()
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_head_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))







