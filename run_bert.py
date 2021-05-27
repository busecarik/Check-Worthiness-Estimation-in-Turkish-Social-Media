import argparse
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, confusion_matrix, classification_report
import neptune
import random

from data_module import BertDataModule
from scorer.subtask_1a import evaluate
from format_checker.subtask_1a import check_format

from transformers import (
    AdamW,
    AutoConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)

NEPTUNE_API = ''

logging.basicConfig(filename='log.txt',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def sigmoid(x): 
  return 1.0 / (1 + np.exp(-x))

def calculate_probs(probs):
    probs_sigmoided = []
    for i in range (len(probs)):
        prob0=sigmoid(probs[i][0])
        prob1=sigmoid(probs[i][1])

        norm_prob0 = prob0 / (prob0 + prob1)
        norm_prob1 = prob1 / (prob0 + prob1)

        if norm_prob0 > norm_prob1:
            probs_sigmoided.append(1 - norm_prob0)
        elif(norm_prob1 >= norm_prob0):
            probs_sigmoided.append(norm_prob1)
    return probs_sigmoided

def compute_metrics(preds, labels):
    labels = [item for sublist in labels for item in sublist]
    preds = [item for sublist in preds for item in sublist]
    f1 = f1_score(labels, preds)
    
    logger.info("F1 score: %.4f%%" % f1)
    logger.info("Precision:  %.4f%%" % precision_score(labels, preds))
    logger.info("Recall:  %.4f%%" % recall_score(labels, preds))
    logger.info(classification_report(labels, preds))
    logger.info("Confusion matrix")
    logger.info(confusion_matrix(labels, preds))
    return f1

def predict(args, model, eval_dataloader):
    probs = []
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)

        with torch.no_grad(): 
            outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
        logits = outputs[0].detach().cpu()
        probs.append(logits.tolist())   
    
    probs = [item for sublist in probs for item in sublist]
    probs = calculate_probs(probs)
    return probs

def eval(args, model, eval_dataloader):
    logger.info("***** Running evaluation *****")
    predictions = []
    true_labels = []
    model.eval()
    total_loss = 0

    for step, batch in enumerate(eval_dataloader):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)

        with torch.no_grad(): 
            outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    labels=labels
                )
        total_loss += outputs[0]
        logits = outputs[1].detach().cpu()
        true_labels.append(labels.to('cpu').tolist())
        predictions.append(torch.argmax(logits, axis=1).tolist())
    
    total_loss /= (step + 1)
    f1 = compute_metrics(predictions, true_labels)
    neptune.log_metric('val_loss', total_loss)
    neptune.log_metric('val_F1', f1)

def train(args, model, train_dataloader, val_dataloader=None):
    num_training_samples = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_training_samples
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {num_training_samples}")
    
    for epoch in range(0, args.num_train_epochs):
        logger.info('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))

        train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            labels = batch[2].to(args.device)

            model.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs[0] / args.gradient_accumulation_steps
            
            loss.backward()
            train_loss += loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
        
        neptune.log_metric('train_loss', (train_loss / (step+1)))  

        eval(args, model, val_dataloader)
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, args.model_save_path)

    logger.info("End of training!")

def main(args):
    dataModule = BertDataModule(args.data_dir, args.model_name_or_path, args.max_seq_len, args.batch_size, args.urlExpand)
    dataModule.load_datasets(args.urlExpand)

    train_dataloader = dataModule.train_dataloader()
    val_dataloader = dataModule.val_dataloader()
    test_dataloader = dataModule.test_dataloader()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    PARAMS = {
        'lr': args.learning_rate,
        'batch size': args.batch_size,
        'model': args. model_name_or_path,
        'epochs': args.num_train_epochs,
        'seed': args.seed,
    }
    neptune.init(api_token=NEPTUNE_API, project_qualified_name='busecarik/Clef')
    neptune.create_experiment('clef_berturk', params=PARAMS)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config).to(args.device)

    if args.do_train:
        train(args, model, train_dataloader, val_dataloader)

    if args.do_eval:
        checkpoint = torch.load(args.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        val_probs = predict(args, model, val_dataloader)

        tweets = pd.read_csv(args.dev_path, sep='\t')
        tweet_ids = tweets['tweet_id'].tolist()

        with open(args.val_output_file, 'w') as probs_file:
            for pred, t in zip(val_probs, tweet_ids):
                probs_file.write("{}\t{}\t{}\t{}\n".format('miscellaneous', str(t), 
                        str(pred), 'berturk'))

        if check_format(args.val_output_file):
            thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(args.dev_path, args.val_output_file)
            logger.info("Val AVGP: {}".format(avg_precision))
            logger.info("Val Reciprocal_rank: {}".format(reciprocal_rank))
            logger.info("Val RP: {}".format(precisions[num_relevant-1]))

    if args.do_predict:
        probs = predict(args, model, test_dataloader)
        with open(args.output_file, 'w') as probs_file:
            for pred, t in zip(probs, tweet_ids):
                probs_file.write("{}\t{}\t{}\t{}\n".format('miscellaneous', str(t), 
                        str(pred), 'berturk'))

        if check_format(args.output_file):
            logger.info("Predictions are printed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--dev_path", default="data/dataset_dev_v1_turkish.tsv", type=str)
    parser.add_argument("--model_save_path", default='cached_models/berturk.pth', type=str)
    parser.add_argument("--do_train", action='store_false', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_false', help="Whether to run evaluation on dev set.")
    parser.add_argument("--do_predict", action='store_false', help="Whether to output predictions on test set.")
    parser.add_argument("--seed", default=12, type=int, help='random seed')
    parser.add_argument("--output_file", type=str, default="outputs/berturk.tsv")
    parser.add_argument("--val_output_file", type=str, default="outputs/berturk.tsv")
    parser.add_argument("--model_name_or_path", type=str, default='dbmdz/bert-base-turkish-cased')
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--max_seq_len", type=int, default=170)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--adam_epsilon", default=1e-8, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--urlExpand", action="store_true", help="Whether change the URL link with domain name")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(args)
    main(args) 