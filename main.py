# -*- encoding: utf-8 -*-
import os
import numpy as np
import random
import argparse
import torch
import logging

from torch import nn
from torch.utils import data
from tqdm import trange
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from data_utils import IncongruityIterableDataset, DataType, flat_accuracy
from bert_pool import BertPoolForIncongruity


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
    exp_dir = os.path.join(args.output_dir, args.exp_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, "logs.txt")
    model_path = os.path.join(exp_dir, args.model_file)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger("bert_incongruity")

    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    s_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    # Set seed
    set_seed(args)

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.max_epochs
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = BertPoolForIncongruity('bert-base-uncased', hidden_size=args.hidden_dim)
    model.cuda()

    if args.freeze:
        model.freeze_bert_encoder()
    else:
        model.unfreeze_bert_encoder()

    # cannot shuffle with iterable dataset
    test_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len, data_type=DataType.Dev)
    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size)

    if args.mode == "train":

        # tokenizer, max_seq_len, filename
        training_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len, data_type=DataType.Train)
        train_dataloader = data.DataLoader(training_set, batch_size=args.batch_size)

        dev_set = IncongruityIterableDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len, data_type=DataType.Dev)
        dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size)

        # Define optimizers
        optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.num_warmup_steps, t_total=args.num_total_steps)
        loss_fct = nn.BCEWithLogitsLoss()
        train_loss_set = []

        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device, dtype=torch.long) for t in batch)
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
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_head_input_ids.size(0)
                nb_tr_steps += 1

            logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))

            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in dev_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device, dtype=torch.long) for t in batch)
                # Unpack the inputs from our dataloader
                b_head_input_ids, b_body_input_ids, b_head_token_type_ids, b_body_token_type_ids, labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = model(b_head_input_ids, b_body_input_ids, b_head_token_type_ids, b_body_token_type_ids)

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            logger.info("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

        torch.save(model.state_dict(), model_path)

    elif args.mode == "test":
        model = torch.load_state_dict(torch.load(model_path))

    else:
        logging.error("Wrong mode: {}".format(args.mode))
        raise TypeError("args.model should be train or test.")

    # Tracking variables
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    # Evaluate test data for one epoch
    for batch in test_dataloader:
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

        tmp_test_accuracy = flat_accuracy(logits, label_ids)

        test_accuracy += tmp_test_accuracy
        nb_test_steps += 1

    logger.info("Test Accuracy: {}".format(test_accuracy / nb_test_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="model: train / test")
    parser.add_argument("--model_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--exp_id", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--output_dir", default="output/", type=str, help="root directory for output")
    parser.add_argument("--seed", default=False, type=float, help="floating value for random seed")
    parser.add_argument("--freeze", default=False, type=bool, help="whether bert parameters are freezed")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max values for gradient clipping")
    parser.add_argument("--num_total_steps", default=1000, type=int, help="For AdamW Secheduler")
    parser.add_argument("--num_warmup_steps", default=100, type=int, help="For AdamW Secheduler")
    parser.add_argument("--max_seq_len", default=512, type=int, help="For AdamW Secheduler")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=2, type=int, help="Number of max epochs for training. btw 2 and 4 are recommended.")
    parser.add_argument("--hidden_dim", default=768, type=int, help="Hidden dims for headline and body text")
    parser.add_argument("--gpu_id", default=2, type=int, help="cuda device index")

    args = parser.parse_args()
    main(args)









