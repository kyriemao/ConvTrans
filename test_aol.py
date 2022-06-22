from IPython import embed
import logging

from pkg_resources import evaluate_marker
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
# from tensorboardX import SummaryWriter

from models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed
from data_structure import ConvDataset
import pytrec_eval

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def test(args):
    eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpu
    # load model
    query_tokenizer, query_encoder = load_model(args.model_type + "_Query", args.query_encoder_checkpoint)
    passage_tokenizer, passage_encoder = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)
    query_encoder = query_encoder.to(args.device)
    passage_encoder = passage_encoder.to(args.device)

    test_dataset = ConvDataset(args, query_tokenizer, passage_tokenizer, args.test_file_path)
    test_loader = DataLoader(test_dataset, 
                            batch_size = eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args, add_doc_info = True, mode = "test_aol"))

    qrels = {}
    run = {}

    num_batch = len(test_dataset) // eval_batch_size
    logger.info("Begin AOL testing...")
    logger.info("The number of total test batches = {}".format(num_batch))
    verbose_num_batch = max(1, int(num_batch / 100))
    cur_batch = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, disable = args.disable_tqdm):
            query_encoder.eval()
            passage_encoder.eval()
            batch_sample_id = batch["bt_sample_id"]

            # query embs
            if args.test_type == "baseline_raw":
                input_ids = batch["bt_raw_query"].to(args.device)
                input_ids_mask = batch["bt_raw_query_mask"].to(args.device)
            elif args.test_type == "conv":
                input_ids = batch["bt_conv_query"].to(args.device)
                input_ids_mask = batch["bt_conv_query_mask"].to(args.device)
            else:
                raise ValueError
           
            # doc embs
            bt_doc = batch["bt_doc"].to(args.device)
            bt_doc_mask = batch["bt_doc_mask"].to(args.device)
            
            seq_len = bt_doc.size(2)
            query_embs = query_encoder(input_ids, input_ids_mask).unsqueeze(1)
            passage_embs = passage_encoder(bt_doc.view(-1, seq_len), bt_doc_mask.view(-1, seq_len))
            passage_embs = passage_embs.view(len(batch_sample_id), -1, query_embs.size(-1))
            scores = torch.sum(query_embs * passage_embs, dim = -1).detach().cpu()
            scores += batch["bt_retrieval_score_mask"].squeeze(-1)

            for i, sample_id in enumerate(batch_sample_id):
                qrels[sample_id] = {}
                run[sample_id] = {}
                for pid in range(batch["bt_pos_label_num"][i]):
                    qrels[sample_id][str(pid)] = 1
                
                for j in range(len(scores[i])):
                    run[sample_id][str(j)] = scores[i][j].item()

            if cur_batch % verbose_num_batch == 0:
                logger.info("{} test batches ok".format(cur_batch))

            cur_batch += 1

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "ndcg_cut.1,3,5,10"})
    res = evaluator.evaluate(run)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    ndcg_1_list = [v['ndcg_cut_1'] for v in res.values()]
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res.values()]
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]
    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@1": np.average(ndcg_1_list),
            "NDCG@3": np.average(ndcg_3_list), 
            "NDCG@5": np.average(ndcg_5_list), 
            "NDCG@10": np.average(ndcg_10_list)}
    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res
    

            
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    test(args)


