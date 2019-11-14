# -*- encoding: utf-8 -*-
import torch
from tqdm import trange
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule

from bert_pool import BertPoolForIncongruity
from util import flat_accuracy


def train_bert_pool():



def main():
    # Parameters:
    lr = 1e-3
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print(sentences[0])
    MAX_LEN = 128

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



    model = BertPoolForIncongruity('bert-base-uncased', do_lower_case=True, hidden_dim=)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    loss_fct = torch.nn.BCEWithLogitsLoss()
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
            logits = model.forward(b_head_input_ids, b_body_input_ids, b_head_token_type_ids, b_body_token_type_ids)
            loss = loss_fct(logits.view(-1, 1), b_labels.view(-1, 1))
            train_loss_set.append(loss.item())

            # Backward pass
            loss.backward()

            # Update parameters and take a step using the computed gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
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
        for batch in validation_dataloader:
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




    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




