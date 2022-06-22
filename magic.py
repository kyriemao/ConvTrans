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
import csv
import argparse
import toml
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer
from data_structure import ConvDataset



def save_model(args, model, query_tokenizer, save_model_order, epoch, step):
    output_dir = oj(args.model_output_path, 'model-{}-epoch-{}-step-{}'.format(save_model_order, epoch, step))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))




def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)



def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)

    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss


def train(args, log_writer):
    # load conversational query encoder model and oracle query model
    query_tokenizer, query_encoder = load_model(args.model_type + "_Query", args.pretrained_query_encoder)
    query_encoder.to(args.device)
    
    if args.train_type == "ranking":
        passage_tokenizer, passage_encoder = load_model(args.model_type + "_Passage", 
                                                        args.pretrained_passage_encoder)
        passage_encoder = passage_encoder.to(args.device)
    else:
        passage_tokenizer = None    


    _, oracle_query_encoder = load_model(args.model_type + "_Query", args.pretrained_query_encoder)
    oracle_query_encoder.to(args.device)

    if args.n_gpu > 1:
        # query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))
        query_encoder = DDP(query_encoder, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    dist.barrier()




    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    train_dataset = ConvDataset(args, query_tokenizer, passage_tokenizer, args.train_file_path, add_doc_info=int(args.train_type == "ranking"))
    ddp_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size, 
                                sampler=ddp_sampler, 
                                collate_fn=train_dataset.get_collate_fn(args, add_doc_info=int(args.train_type == "ranking"), mode="train_msmarco"))
    
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.save_steps, float):
        args.save_steps = int(args.save_steps * num_steps_per_epoch)
        args.save_steps = max(1, args.save_steps)
    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
    for epoch in epoch_iterator:
        query_encoder.train()
        oracle_query_encoder.eval()
        
        train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()

            bt_conv_query = batch['bt_conv_query'].to(args.device) 
            bt_conv_query_mask = batch['bt_conv_query_mask'].to(args.device)
            conv_query_embs = query_encoder(bt_conv_query, bt_conv_query_mask)  # B * dim
            

            if args.train_type == "kd":
                bt_oracle_query = batch["bt_oracle_query"].to(args.device)
                bt_oracle_query_mask = batch["bt_oracle_query_mask"].to(args.device)
                with torch.no_grad():
                # freeze oracle query encoder's parameters
                    oracle_query_embs = oracle_query_encoder(bt_oracle_query, bt_oracle_query_mask).detach()  # B * dim
                loss = cal_kd_loss(conv_query_embs, oracle_query_embs)

            elif args.train_type == "ranking":
                bt_pos_docs = batch['bt_pos_docs'].to(args.device)
                bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
                bt_neg_docs = batch['bt_neg_docs'].to(args.device)
                bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
                with torch.no_grad():
                # freeze passage encoder's parameters
                    pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                    neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim, hard negative
                loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            if dist.get_rank() == 0 and args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, Loss = {}".format(
                                epoch,
                                global_step,
                                loss.item())
                            )

            if dist.get_rank() == 0:
                log_writer.add_scalar("train_{}_loss".format(args.train_type), loss, global_step)
            
            global_step += 1    # avoid saving the model of the first step.
            dist.barrier()
            
            # save model finally
            if dist.get_rank() == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, query_encoder, query_tokenizer, save_model_order, epoch, global_step)
                save_model_order += 1


    logger.info("Training finish!")          
         


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work

    args = parser.parse_args()
    config = toml.load(args.config)
    local_rank = args.local_rank
    args = argparse.Namespace(**config)
    args.local_rank = local_rank

    # pytorch parallel gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    args.device = device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    
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
    
    if dist.get_rank() == 0:    
        log_writer = SummaryWriter(log_dir = args.log_dir_path)
    else:
        log_writer = None
    train(args, log_writer)
    log_writer.close()

#  python pretrain_msmarco_rule.py --config Config/pretrain_msmarco_rule.toml 