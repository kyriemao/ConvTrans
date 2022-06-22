from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def main():
    args = get_args()
    
    query_list = []
    qid_list = []

    max_passage_token_length = 512

    with open(args.input_query_path, "r") as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line)
        if args.query_type == "raw":
            query = line["query"]
        elif args.query_type == "oracle":
            query = line['oracle_query']
        elif args.query_type == "concat":
            query = " ".join(line['context_queries'] + [line['query']])
        elif args.query_type == "concat_with_last_answer":
            query = " ".join(line['context_queries'] + [line["last_answer"]] + [line['query']])
        elif args.query_type == "concat_with_last_passage":
            last_response = line["last_response"].split(' ')[:max_passage_token_length]
            last_response = ' '.join(last_response)
            query = " ".join(line['context_queries'] + [last_response] + [line['query']])

        query_list.append(query)
        qid_list.append(line['sample_id'])


    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 20)

    
    with open(oj(args.output_dir_path, "res.trec"), "w") as f:
        for qid in qid_list:
            if qid not in hits:
                print("{} not in hits".format(qid))
                continue
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid,
                                                i+1,
                                                -i - 1 + 200,
                                                "bm25"
                                                ))
                f.write('\n')


    res = print_res(oj(args.output_dir_path, "res.trec"), args.gold_qrel_file_path, args.rel_threshold)
    return res



def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.split("\t")
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in qrels:
            continue
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
            "NDCG@3": np.average(ndcg_3_list), 
        }

    
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

    check_dir_exist_or_build([args.output_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args




if __name__ == '__main__':
    main()

