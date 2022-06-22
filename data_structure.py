# data structure library file

from multiprocessing.sharedctypes import Value
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class ConvExample:
    def __init__(self, sample_id, 
                       conv_query, 
                       cur_query_position = -1, 
                       pos_docs = None,
                       neg_docs = None,
                       raw_query = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.conv_query = conv_query
        self.cur_query_position = cur_query_position
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs

        self.raw_query = raw_query
        self.oracle_query = oracle_query




class ConvDataset(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []

        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['sample_id']
           
            conv_query, query, cur_query_position = self.build_conv_query(data[i], query_tokenizer, args, concat_order="reverse")

            # doc info for ranking loss
            pos_docs = []
            neg_docs = []
            if add_doc_info:
                for doc in data[i]['pos_docs']:
                    pos_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))
                seen_neg_docs = set()
                for doc in data[i]['neg_docs']:
                    if doc in data[i]['pos_docs'] or doc in seen_neg_docs:
                        continue
                    seen_neg_docs.add(doc)
                    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))
                    if args.only_one_negative:
                        break   
                # if no valid pos_docs or neg_docs, skip this sample
                if len(pos_docs) == 0 or len(neg_docs) == 0:
                    continue

            # For baseline test
            raw_query = query
            if "oracle_query" in data[i]: 
                oracle_query = query_tokenizer.encode(data[i]['oracle_query'] , add_special_tokens=True, max_length=args.max_query_length)
            else:
                oracle_query = None

            self.examples.append(ConvExample(sample_id, 
                                             conv_query, 
                                             cur_query_position, 
                                             pos_docs, 
                                             neg_docs,
                                             raw_query = raw_query,
                                             oracle_query = oracle_query))            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    def build_conv_query(self, sample, query_tokenizer, args, concat_order = "reverse"):
        # current query
        query = query_tokenizer.encode(sample['query'] , add_special_tokens=True, max_length=args.max_query_length)
        if args.special_assemble_conv_query:
            query = query[1:]

        # context
        context = []
        context_queries = sample['context_queries']
        if concat_order == "reverse":
            for i in range(len(context_queries) - 1, -1, -1):
                cq = context_queries[i]
                cq = query_tokenizer.encode(cq, add_special_tokens=True, max_length=args.max_query_length)
                if args.special_assemble_conv_query:
                    cq = cq[1:]         
                context.extend(cq)
        else:
            for cq in context_queries:     
                cq = query_tokenizer.encode(cq, add_special_tokens=True, max_length=args.max_query_length)
                if args.special_assemble_conv_query:
                    cq = cq[1:]         
                context.extend(cq)
    
        # last response
        last_response = []
        if args.enable_last_repsone:
            if not args.special_assemble_conv_query:
                last_response.append(query_tokenizer.cls_token_id)
            last_response.extend(query_tokenizer.convert_tokens_to_ids(["<CTX_R>"]))
            if args.response_type == "answer":
                if "last_answer" in sample and sample['last_answer']:
                    last_response.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(sample['last_answer'])))
            elif args.response_type == "passage":
                if "last_response" in sample and sample['last_response']:
                    last_response.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(sample['last_response'])))
            else:
                raise ValueError
            last_response = last_response[:args.max_doc_length]
            last_response.append(query_tokenizer.sep_token_id)
        
        # assemble them
        if args.special_assemble_conv_query:
            conv_query = [query_tokenizer.cls_token_id] # CLS CUR_Q q SEP CTX CTX_R response SEP CTX_Q cq1 SEP cq2 SEP ... SEP
            conv_query.extend(query_tokenizer.convert_tokens_to_ids(["<CUR_Q>"]))
            conv_query += query
            conv_query.extend(query_tokenizer.convert_tokens_to_ids(["<CTX>"]))
            conv_query += last_response
            conv_query.extend(query_tokenizer.convert_tokens_to_ids(["<CTX_Q>"]))
            conv_query += context
            cur_query_position = 0
        else:
            if concat_order == "reverse":
                conv_query = query + last_response + context
                cur_query_position = 0
            else:
                conv_query = context + last_response + query
                cur_query_position = len(context) + len(last_response)
        
        if args.special_assemble_conv_query:
            query = [query_tokenizer.cls_token_id] + query

        return conv_query, query, cur_query_position




        
    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_cur_query_position":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_cur_query_position = []

            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []


            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)
     
                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_cur_query_position.append(example.cur_query_position)
                
                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    neg_doc = random.sample(example.neg_docs, 1)[0] # BM25 hard negative or random
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_concat_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_concat_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

                
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_cur_query_position"] = bt_cur_query_position
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
             
            return collated_dict
        

        def collate_fn_test_aol(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_cur_query_position":[],
                "bt_doc":[],
                "bt_doc_mask":[],
                "bt_pos_label_num":[],
                "bt_retrieval_score_mask":[],
            }

            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_cur_query_position = []


            # for doc
            bt_doc = []
            bt_doc_mask = []
            bt_pos_label_num = []
            bt_retrieval_score_mask = []

            bt_raw_query = []
            bt_raw_query_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                    
                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_cur_query_position.append(example.cur_query_position)
                
                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)

                docs = []
                docs_mask = []
                pos_label_num = 0
                for pos_doc in example.pos_docs:
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_concat_length)
                    docs.append(pos_doc)
                    docs_mask.append(pos_doc_mask)
                    pos_label_num += 1
                for neg_doc in example.neg_docs:
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_concat_length)
                    docs.append(neg_doc)
                    docs_mask.append(neg_doc_mask)
            
                
                bt_doc.append(torch.tensor(docs))
                bt_doc_mask.append(torch.tensor(docs_mask))
                bt_pos_label_num.append(pos_label_num)
                bt_retrieval_score_mask.append(torch.zeros((len(docs), 1)))
            
            # pad doc number
    
            bt_doc = pad_sequence(bt_doc, batch_first = True)   # B * max_doc_num * seq_len
            bt_doc_mask = pad_sequence(bt_doc_mask, batch_first = True)
            bt_retrieval_score_mask = pad_sequence(bt_retrieval_score_mask, batch_first=True, padding_value=-np.inf)
            
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_cur_query_position"] = bt_cur_query_position
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask

            collated_dict["bt_doc"] = bt_doc
            collated_dict["bt_doc_mask"] = bt_doc_mask
            collated_dict["bt_retrieval_score_mask"] = bt_retrieval_score_mask
            collated_dict["bt_pos_label_num"] = bt_pos_label_num
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_doc", "bt_doc_mask", "bt_pos_label_num", "bt_retrieval_score_mask"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict


        if mode == "test_aol":
            return collate_fn_test_aol
        elif mode in ["train_aol", "train_cast", "test_cast", "train_msmarco"]:
            return collate_fn
        else:
            raise ValueError


class Session2ConvSample:
    def __init__(self, sample_id, conv_input, labels, pred_begin_pos):
        self.sample_id = sample_id
        self.conv_input = conv_input
        self.labels = labels
        self.pred_begin_pos = pred_begin_pos
    

class Session2ConvDataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)

            kw_query = record['kw_query']
            context_kw_queries = record['context_kw_queries']
            context_kw_queries = []
            context_oracle_queries = record['context_oracle_queries']
            context_oracle_queries = []
            # last_response = record["last_response"]
            oracle_query = record["oracle_query"]
            query = record['query'] # target

            kw_conv_input = []
            oracle_conv_input = []
            kw_labels = []
            oracle_labels = []
            
            # covn_input format: context + kw_query/oracle_query
            # target output: query
            # so one turn can build out two examples
            # need to be refined
            
            # context
            assert len(context_kw_queries) == len(context_oracle_queries)
            if len(context_kw_queries) > 0:
                for turn_idx, sent in enumerate(context_kw_queries):
                    kw_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                    kw_conv_input.append(tokenizer.sep_token_id)
                    # oracle_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context_oracle_queries[turn_idx])))
                    # oracle_conv_input.append(tokenizer.sep_token_id)

                max_seq_length = args.max_seq_length - 30
                if len(kw_conv_input) > max_seq_length:
                    kw_conv_input = kw_conv_input[:max_seq_length]
                    kw_conv_input.append(tokenizer.sep_token_id)
                # if len(oracle_conv_input) > max_seq_length:
                #     oracle_conv_input = oracle_conv_input[:max_seq_length]
                #     oracle_conv_input.append(tokenizer.sep_token_id)

            # current kw_query/oracle_query
            kw_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(kw_query)))
            kw_conv_input.append(tokenizer.bos_token_id)
            # oracle_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(oracle_query)))
            # oracle_conv_input.append(tokenizer.bos_token_id)

            # target: query
            kw_pred_begin_pos = len(kw_conv_input)
            kw_labels.extend([-100] * kw_pred_begin_pos)
            kw_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(oracle_query)))
            kw_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(oracle_query)))
            kw_conv_input.append(tokenizer.eos_token_id)
            kw_labels.append(tokenizer.eos_token_id)
            
            # oracle_pred_begin_pos = len(oracle_conv_input)
            # oracle_labels.extend([-100] * oracle_pred_begin_pos)
            # oracle_conv_input.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query)))
            # oracle_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query)))
            # oracle_conv_input.append(tokenizer.eos_token_id)
            # oracle_labels.append(tokenizer.eos_token_id)

            # padding
            if len(kw_conv_input) <= args.max_seq_length:   # this sample is too long, skip it.
                pad_num = args.max_seq_length - len(kw_conv_input)
                kw_conv_input.extend([tokenizer.pad_token_id] * pad_num)
                kw_labels.extend([-100] * pad_num)
                example = Session2ConvSample(record['sample_id'] + "_kw", kw_conv_input, kw_labels, kw_pred_begin_pos)
                self.examples.append(example)

            # if len(oracle_conv_input) <= args.max_seq_length:   # this sample is too long, skip it.
            #     pad_num = args.max_seq_length - len(oracle_conv_input)
            #     oracle_conv_input.extend([tokenizer.pad_token_id] * pad_num)
            #     oracle_labels.extend([-100] * pad_num)
            #     example = Session2ConvSample(record['sample_id'] + "_oracle", oracle_conv_input, oracle_labels, oracle_pred_begin_pos)
            #     self.examples.append(example)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


    @staticmethod
    def get_collate_fn():

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_input":[],
                "bt_labels":[],
                "bt_pred_begin_pos":[]}
            for example in batch:
                collated_dict["bt_sample_id"].append(example.sample_id)
                collated_dict["bt_conv_input"].append(example.conv_input)
                collated_dict["bt_labels"].append(example.labels)
                collated_dict["bt_pred_begin_pos"].append(example.pred_begin_pos)
            
            collated_dict["bt_conv_input"] = torch.tensor(collated_dict["bt_conv_input"])
            collated_dict["bt_labels"] = torch.tensor(collated_dict["bt_labels"])
            return collated_dict

        return collate_fn


class Simple_Kw_to_Nl_Inference_Sample:
    def __init__(self, qid, query, input_ids, attention_mask):
        self.qid = qid
        self.query = query
        self.input_ids = input_ids
        self.attention_mask = attention_mask

class Simple_Kw_to_Nl_Inference_Dataset:
    def __init__(self, args, qid2query, tokenizer):
        self.examples = []
        for qid in qid2query:
            query = qid2query[qid]
            input_seq = query
            input_encoding = tokenizer(
                input_seq,
                padding="max_length",
                max_length=args.max_kw_to_nl_seq_length,
                truncation=True,
                return_tensors="pt",
            )    
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            example = Simple_Kw_to_Nl_Inference_Sample(qid, query, input_ids, attention_mask)
            self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn():
        def collate_fn(batch: list):
            collated_dict = {"bt_qid":[],
                             "bt_query":[],
                             "bt_input_ids": [],
                             "bt_attention_mask": []}
            for example in batch:
                collated_dict["bt_qid"].append(example.qid)
                collated_dict["bt_query"].append(example.query)
                collated_dict["bt_input_ids"].append(example.input_ids)
                collated_dict["bt_attention_mask"].append(example.attention_mask)

            for key in collated_dict:
                if key in ["bt_qid", "bt_query"]:
                    continue
                collated_dict[key] = torch.cat(collated_dict[key], dim=0)
            
            return collated_dict
        return collate_fn


class Kw_to_Nl_with_T5_Sample:
    def __init__(self, sample_id, kw_query, input_ids, attention_mask, labels):
        self.sample_id = sample_id
        self.kw_query = kw_query
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    

class Kw_to_Nl_with_T5_Dataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        self.task_prefix = "expand: "
        for line in tqdm(data):
            record = json.loads(line)

            kw_query = record['kw_query']   # input
            input_seq = kw_query
            input_encoding = tokenizer(
                self.task_prefix + input_seq if args.enable_task_prefix else input_seq,
                padding="max_length",
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )    
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            if "oracle_query" in record:
                oracle_query = record["oracle_query"]   # target output for training t5
                target_seq = oracle_query
                target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_seq_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.unsqueeze(0)
            else:
                labels = None

            sample_id = "-1_-1" if "sample_id" not in record else record["sample_id"]
            self.examples.append(Kw_to_Nl_with_T5_Sample(sample_id, 
                                                         kw_query, 
                                                         input_ids, 
                                                         attention_mask, 
                                                         labels))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(mode="train"):

        def train_collate_fn(batch: list):
            collated_dict = {"bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": []
                            }
            for example in batch:
                collated_dict["bt_input_ids"].append(example.input_ids)
                collated_dict["bt_attention_mask"].append(example.attention_mask)
                collated_dict["bt_labels"].append(example.labels)

            for key in collated_dict:
                collated_dict[key] = torch.cat(collated_dict[key], dim=0)
            
            return collated_dict

        def test_collate_fn(batch: list):
            collated_dict = {"bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_sample_id": [],
                             "bt_kw_query":[]
                            }
            for example in batch:
                collated_dict["bt_input_ids"].append(example.input_ids)
                collated_dict["bt_attention_mask"].append(example.attention_mask)
                collated_dict["bt_sample_id"].append(example.sample_id)
                collated_dict["bt_kw_query"].append(example.kw_query)

            for key in collated_dict:
                if key in ["bt_input_ids", "bt_attention_mask"]:
                    collated_dict[key] = torch.cat(collated_dict[key], dim=0)

            return collated_dict
        
        if mode == "train":
            return train_collate_fn
        elif mode == "test":
            return test_collate_fn
        else:
            raise NotImplementedError




class Nl_to_Cnl_with_T5_Sample:
    def __init__(self, sample_id, query, oracle_query, context, input_ids, attention_mask, labels):
        self.sample_id = sample_id
        self.query = query
        self.oracle_query = oracle_query
        self.context = context
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class Nl_to_Cnl_with_T5_Dataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)

            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            conv_input_seq = "<CUR>"
            oracle_query = record["oracle_query"]
            context = record["context"]
            conv_input_seq += " " + oracle_query
            conv_input_seq += " " + "<CTX>"
            for i in range(len(context)):
                conv_input_seq += " " + context[i]
                if i > 0:
                    conv_input_seq += " " + "<SEP>"

            input_encoding = tokenizer(
                conv_input_seq,
                padding="max_length",
                max_length=args.max_nl_to_cnl_input_seq_length,
                truncation=True,
                return_tensors="pt",
            )    
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            if "query" in record:
                query = record["query"]   # target output for training t5 for nl to cnl
                target_seq = query
                target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_nl_to_cnl_output_seq_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.unsqueeze(0)
            else:
                query = None
                labels = None

            sample_id = "-1_-1" if "sample_id" not in record else record["sample_id"]
            self.examples.append(Nl_to_Cnl_with_T5_Sample(sample_id, 
                                                         query,
                                                         oracle_query,
                                                         context, 
                                                         input_ids, 
                                                         attention_mask, 
                                                         labels))
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(mode="train"):

        def train_collate_fn(batch: list):
            collated_dict = {"bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": []
                            }
            for example in batch:
                collated_dict["bt_input_ids"].append(example.input_ids)
                collated_dict["bt_attention_mask"].append(example.attention_mask)
                collated_dict["bt_labels"].append(example.labels)

            for key in collated_dict:
                collated_dict[key] = torch.cat(collated_dict[key], dim=0)
            
            return collated_dict

        def test_collate_fn(batch: list):
            collated_dict = {"bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_sample_id": [],
                             "bt_oracle_query":[],
                             "bt_context":[]
                            }
            for example in batch:
                collated_dict["bt_input_ids"].append(example.input_ids)
                collated_dict["bt_attention_mask"].append(example.attention_mask)
                collated_dict["bt_sample_id"].append(example.sample_id)
                collated_dict["bt_oracle_query"].append(example.oracle_query)
                collated_dict["bt_context"].append(example.context)

            for key in collated_dict:
                if key in ["bt_input_ids", "bt_attention_mask"]:
                    collated_dict[key] = torch.cat(collated_dict[key], dim=0)

            return collated_dict
        
        if mode == "train":
            return train_collate_fn
        elif mode == "test":
            return test_collate_fn
        else:
            raise NotImplementedError





def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0,
                            concat_order = "reverse"):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        if concat_order == "reverse":
            input_ids = input_ids[:max_length]
        else:
            input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask

