from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import time
import copy
import pickle
import random
import numpy as np
import argparse
import toml
import os
from os import path
from os.path import join as oj
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tensorboardX import SummaryWriter

from models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer
from data_structure import ConvDataset, Session2ConvDataset

special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}



def save_model(args, model, tokenizer, save_model_order, epoch, step):
    output_dir = oj(args.model_output_path, 'model-{}-epoch-{}-step-{}'.format(save_model_order, epoch, step))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))


def train_gpt(args, log_writer):
    # load gpt model and resize the tokenizer
    config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    config = config_class.from_pretrained(args.model_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    tokenizer.add_special_tokens(special_tokens_dict)
    model = model_class.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))  # resize
    model.to(args.device)
    
    # training data
    train_dataset = Session2ConvDataset(args, tokenizer, args.train_file_path)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=train_sampler, 
                                  batch_size=args.batch_size, 
                                  collate_fn=train_dataset.get_collate_fn())

    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))    

    # Prepare optimizer and schedule (linear warmup and decay)
    # This is specific for gpt training
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    if isinstance(args.save_steps, float):
        args.save_steps = int(args.save_steps * num_steps_per_epoch)
        args.save_steps = max(1, args.save_steps)
    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", total_training_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
    global_step = 0
    save_model_order = 0

    for epoch in epoch_iterator:
        for batch in tqdm(train_dataloader,  desc="Step", disable=args.disable_tqdm):
            model.zero_grad()

            inputs, labels = (batch["bt_conv_input"], batch["bt_labels"])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, Train Loss = {}".format(
                                epoch,
                                global_step,
                                loss.item())
                            )
            log_writer.add_scalar("train_gpt_loss", loss, global_step)
            
            global_step += 1    # avoid saving the model of the first step.
            # save model finally
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, tokenizer, save_model_order, epoch, global_step)
                save_model_order += 1


    logger.info("Training finish!")




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    # device + dir check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if os.path.exists(args.model_output_path) and os.listdir(
        args.model_output_path) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.model_output_path))

    check_dir_exist_or_build([args.model_output_path, args.log_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
  
    log_writer = SummaryWriter(log_dir = args.log_dir_path)
    train_gpt(args, log_writer)
    log_writer.close()