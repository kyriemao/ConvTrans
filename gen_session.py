from multiprocessing.sharedctypes import Value
from nis import match
from IPython import embed
import logging

from requests import session
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import argparse
import toml
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
import re
import random
import gc
import nltk
from nltk.corpus import wordnet 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import heapq
import torch
from torch.utils.data import DataLoader
from utils import BIG_STOP_WORDS, pload, pstore, QUESTION_WORD_LIST,format_nl_query, is_nl_query
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_structure import Simple_Kw_to_Nl_Inference_Dataset

# nltk.download('wordnet')


def query_formatting(q):
    q = re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "", q)

    q = q.lower()

    q = q.split(" ")
    for i in range(len(q)):
        q[i] = q[i].replace(" ", "")
    q = " ".join(q)
    q = q.rstrip().lstrip()
    
    return q

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_all_qrels(args):
    # get qrels
    with open(args.passage_file, "r") as f:
        passages = f.readlines()
    print("load passage ok")

    qrels = {}
    for qrel_file in args.qrel_files:
        with open(qrel_file, "r") as f:
            data = f.readlines()

        for line in tqdm(data):
            line = line.strip().split('\t')
            qid = int(line[0])
            pid = int(line[2])
            if int(line[3]) < 1:
                continue
            try:
                pid, passage = passages[pid].strip().split('\t')
                pid = int(pid)
            except:
                continue
            qrels[qid] = (pid, passage)
    
    pstore(qrels, "qrels.pkl")
    return qrels

def get_response_induced_next_query_dict(args):
    qid2query = pload("qid2query.pkl")
    qrels = pload("qrels.pkl")
    query2qid = {qid2query[qid]:qid for qid in qid2query}

    pid2nextqid = {}
    for file in args.wss_files:
        with open(file, "r") as fin:
            for line in tqdm(fin):
                line = line.strip().split('\t')
                last_pid = -1
                for query in line[1:]:
                    query = query_formatting(query)
                    if query not in query2qid:
                        last_pid = -1
                        continue
                    qid = query2qid[query]
                    if qid not in qrels:
                        last_pid = -1
                        continue
                    if last_pid != -1:
                        if last_pid not in pid2nextqid:
                            pid2nextqid[last_pid] = set()
                        pid2nextqid[last_pid].add(qid)
                    last_pid = qrels[qid][0]
    
    pstore(pid2nextqid, "pid2nextqid.pkl")

    return pid2nextqid
                    

def query_formatting_and_extract_topic_words(args):
    qid2query = {}
    qid2tw = {}

    lemmatizer = WordNetLemmatizer()
    stop_words_set = set(BIG_STOP_WORDS)
    
    for query_file in args.query_files:
        with open(query_file, "r") as f:
            data = f.readlines()
   
        for line in tqdm(data):
            line = line.strip().split('\t')
            qid = int(line[0])
            query = query_formatting(line[1])
            if qid not in qid2query:
                qid2query[qid] = query
                tw_set = set()
                text = nltk.word_tokenize(query)
                tagged_text = nltk.pos_tag(text)
                for token in tagged_text:
                    if token[0] in stop_words_set:
                        continue
                    tag = get_wordnet_pos(token[1])                    
                    if tag is None:
                        tw = lemmatizer.lemmatize(token[0])
                    else:
                        tw = lemmatizer.lemmatize(token[0], tag)
                    tw_set.add(tw)
                qid2tw[qid] = tw_set
            else:
                print(qid)
                print(qid2query[qid])
                raise ValueError
    

    # query2qid = {qid2query[qid]:qid for qid in qid2query}

    # build tw2query inverted index
    tw2qid = {}
    for qid in tqdm(qid2tw):
        for word in qid2tw[qid]:
            if word not in tw2qid:
                tw2qid[word] = set()
            tw2qid[word].add(qid)
    
    
    pstore(qid2query, "qid2query.pkl")
    pstore(qid2tw, "qid2tw.pkl")
    pstore(tw2qid, "tw2qid.pkl")
    print("qid2query, qid2tw, tw2qid, store ok")
    
    return qid2query, qid2tw, tw2qid 

def gen_new_session_graph(orig_session_id, 
                          session_queries, 
                          session_qids, 
                          qid2query, 
                          query2qid, 
                          qid2tw, 
                          tw2qid, 
                          qrels, 
                          pid2nextqid, 
                          kw_model):
    used_qid_set = set()
    new_turns = []
    lemmatizer = WordNetLemmatizer()

    for turn_idx, this_query in enumerate(session_queries):
        # per turn
        this_qid = query2qid[this_query]
        if this_qid in used_qid_set:
            continue
        used_qid_set.add(this_qid)
        this_tw_set = qid2tw[this_qid]

        # RI: response-induced relation
        RI_qid_sent_list = []
        pid, passage = qrels[this_qid]
        sents = re.split('[.?!]', passage)  # split to multiple sentences
        sentid2kw_dict = {}
        for sent_id, sent in enumerate(sents):
            sent = sent.rstrip().lstrip()
            if sent == "":
                continue
            keyword_set = set()
            keywords = kw_model.extract_keywords(sent, keyphrase_ngram_range = (1,2), top_n=6, stop_words=BIG_STOP_WORDS)
            for kw in keywords:
                keyword_set = keyword_set | set(kw[0].split(" ")) - this_tw_set # remove the keywords of the query
            new_keyword_set = set()
            for kw in keyword_set:
                new_keyword_set.add(lemmatizer.lemmatize(kw))
            sentid2kw_dict[sent_id] = new_keyword_set
        # RI from the same session
        for qid in session_qids[turn_idx + 1:]:
            if qid in used_qid_set:
                continue
            tw_set = qid2tw[qid]
            RI_threshold = int(len(tw_set) / 2)  # each query has its RI-threshold
            if RI_threshold <= 0:
                continue
            # only keep one sent for each qid
            final_sent_id, max_match_val = -1, -1
            for sent_id in sentid2kw_dict:
                match_val = len(sentid2kw_dict[sent_id] & tw_set)
                if match_val > RI_threshold:
                    match_val /= len(tw_set)
                    if match_val > max_match_val:
                        max_match_val = match_val
                        final_sent_id = sent_id
            if final_sent_id != -1:
                RI_qid_sent_list.append((qid, sents[final_sent_id], max_match_val))
                used_qid_set.add(qid)
        
        
        if len(RI_qid_sent_list) >= args.max_response_induced_num:
            RI_qid_sent_list = sorted(RI_qid_sent_list, key=lambda x:-x[2])
            RI_qid_sent_list = RI_qid_sent_list[:args.max_response_induced_num]
        else:
            # RI from the out-of sesssion 
            out_session_RI_qid_sent_list = []
            if pid in pid2nextqid:
                candidate_RI_qids = pid2nextqid[pid]
                for qid in candidate_RI_qids:
                    if qid in used_qid_set:
                        continue
                    tw_set = qid2tw[qid]
                    RI_threshold = int(len(tw_set) / 2)  # each query has its RI-threshold
                    if RI_threshold <= 0:
                        continue
                    # only keep one sent for each qid
                    final_sent_id, max_match_val = -1, -1
                    for sent_id in sentid2kw_dict:
                        match_val = len(sentid2kw_dict[sent_id] & tw_set)
                        if match_val > RI_threshold:
                            match_val /= len(tw_set)
                            if match_val > max_match_val:
                                max_match_val = match_val
                                final_sent_id = sent_id
                    if final_sent_id != -1:
                        out_session_RI_qid_sent_list.append((qid, sents[final_sent_id], max_match_val))
                        used_qid_set.add(qid)
        

                out_session_RI_qid_sent_list_top = heapq.nlargest(args.max_response_induced_num - len(RI_qid_sent_list), out_session_RI_qid_sent_list, key=lambda x:x[2])
                RI_qid_sent_list += out_session_RI_qid_sent_list_top

  
        # IT: in-topic relation
        IT_threshold = len(this_tw_set) / 2 # in-topic threshold
        IT_qid_list = []    # in-topic qid list for this turn

        if IT_threshold > 0:     
            # in_topic from the same session
            for qid in session_qids[turn_idx + 1:]:
                if qid in used_qid_set:
                    continue
                intersect_num = len(this_tw_set & qid2tw[qid])
                if intersect_num > IT_threshold:
                    match_val = -intersect_num / len(qid2tw[qid])
                    IT_qid_list.append((qid, match_val))
                    used_qid_set.add(qid)

            if len(IT_qid_list) >= args.max_in_topic_num:
                IT_qid_list = sorted(IT_qid_list, key=lambda x:-x[1])
                IT_qid_list = IT_qid_list[:args.max_in_topic_num]
            else:
                # in_topic from out-of session
                qid2num_dict = {}
                for tw in this_tw_set:
                    qids = tw2qid[tw]
                    for qid in qids:
                        if qid not in qid2num_dict:
                            qid2num_dict[qid] = 0
                        qid2num_dict[qid] += 1

                out_session_in_topic_qid_list = []
                for qid in qid2num_dict:
                    if qid in used_qid_set:
                            continue
                    if qid2num_dict[qid] > IT_threshold:
                        match_val = -qid2num_dict[qid] / len(qid2tw[qid])
                        out_session_in_topic_qid_list.append((qid, match_val))  # (qid, match_val (larger is better))
                        used_qid_set.add(qid)

                out_session_in_topic_qid_list_top = heapq.nlargest(args.max_in_topic_num - len(IT_qid_list), out_session_in_topic_qid_list, key=lambda x:x[1])
                IT_qid_list += out_session_in_topic_qid_list_top
    
        # OI: out-topic relation
        OI_qid = -1
        for idx in range(turn_idx + 1, len(session_qids)):
            qid = session_qids[idx]
            if qid not in used_qid_set:
                OI_qid = qid
                break

        # output
        this_turn = {}
        this_turn["query"] = this_query
        this_turn["qid"] = this_qid
        this_turn["IT_qid_list"] = [x[0] for x in IT_qid_list]
        this_turn["RI_qid_sent_list"] = [(x[0],x[1]) for x in RI_qid_sent_list]
        this_turn["OI_qid"] = OI_qid

        new_turns.append(this_turn)
    

    # check new_session_graph
    for i in range(1, len(new_turns)):
        assert new_turns[i]["qid"] == new_turns[i-1]["OI_qid"]
    seen_qids = {}
    for turn in new_turns:
        qid = turn["qid"]
        if qid not in seen_qids:
            seen_qids[qid] = 1
        else:
            raise ValueError
        
        for qid in turn["IT_qid_list"]:
            if qid not in seen_qids:
                seen_qids[qid] = 1
            else:
                raise ValueError

        for x in turn["RI_qid_sent_list"]:
            qid = x[0]
            if qid not in seen_qids:
                seen_qids[qid] = 1
            else:
                raise ValueError
    

    new_session_graph = {}
    new_session_graph["session_id"] = orig_session_id
    new_session_graph["turns"] = new_turns

    return new_session_graph


def process_session(args):
    qid2query = pload("qid2query.pkl")
    query2qid = {qid2query[qid]:qid for qid in qid2query}
    qid2tw = pload("qid2tw.pkl")
    tw2qid = pload("tw2qid.pkl")
    qrels = pload("qrels.pkl")
    pid2nextqid = pload("pid2nextqid.pkl")

    backend_name = "msmarco-bert-base-dot-v5"
    sentence_model = SentenceTransformer(backend_name)
    kw_model = KeyBERT(sentence_model)

    has_RI_turn_num = 0
    all_turn_num = 0
    n = 0
    with open(args.wss_file, "r") as fin:
        data = fin.readlines()    

    with open(args.wss_meta_graph_file, "w") as fout:
        for line in tqdm(data):
            # per session
            line = line.strip().split("\t")
            orig_session_id = line[0]
            queries = line[1:]
            seen_query_set = set()
            new_queries = []
            new_qids = []
            for query in queries:
                query = query_formatting(query)
                if query in seen_query_set:
                    continue
                qid = query2qid[query]
                if qid not in qrels:
                    continue
                new_queries.append(query)
                new_qids.append(qid)
                seen_query_set.add(query)
         
            new_session_graph = gen_new_session_graph(orig_session_id,
                                                      new_queries, 
                                                      new_qids,
                                                      qid2query, 
                                                      query2qid, 
                                                      qid2tw, 
                                                      tw2qid, 
                                                      qrels, 
                                                      pid2nextqid, 
                                                      kw_model)
            for turn in new_session_graph["turns"]:
                if len(turn["RI_qid_sent_list"]) > 0:
                    has_RI_turn_num += 1
            all_turn_num += len(new_session_graph["turns"])

            fout.write(json.dumps(new_session_graph))
            fout.write('\n')

            n += 1
            if n % 200 == 0:   
                print("has RI turn num: {}".format(has_RI_turn_num))
                print("all_turn_num: {}".format(all_turn_num))

    print("has RI turn num: {}".format(has_RI_turn_num))
    print("all_turn_num: {}".format(all_turn_num))




def process_session_kw_to_nl(args):
    # build qid2kwnl_dict
    qid2query = pload("qid2query.pkl")
    kw_qid2query = {}
    for qid in qid2query:
        query = qid2query[qid]
        if is_nl_query(query):
            continue
        kw_qid2query[qid] = query
    
    tokenizer = T5Tokenizer.from_pretrained(args.kw_to_nl_t5_checkpoint_path)
    model = T5ForConditionalGeneration.from_pretrained(args.kw_to_nl_t5_checkpoint_path)
    model.to(args.device)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    inference_dataset = Simple_Kw_to_Nl_Inference_Dataset(args, kw_qid2query, tokenizer)
    inference_dataloader = DataLoader(inference_dataset, 
                                      shuffle=False,
                                      batch_size=args.inference_kw_to_nl_batch_size,
                                      collate_fn=inference_dataset.get_collate_fn())

    with open(args.kw_to_nl_output_file, "w") as fout:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(inference_dataloader, desc="Step"):
                bt_input_ids = batch["bt_input_ids"].to(args.device)
                bt_attention_mask = batch["bt_attention_mask"].to(args.device)
                output_seqs = model.generate(input_ids=bt_input_ids, 
                                                attention_mask=bt_attention_mask, 
                                                do_sample=False,
                                                max_length=args.max_kw_to_nl_seq_length)
                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
                for i in range(len(outputs)):
                    record = {}
                    record["qid"] = batch["bt_qid"][i]
                    record["kw_query"] = batch["bt_query"][i]
                    record["nl_query"] = format_nl_query(outputs[i])
                    # qid2query[record["qid"]] = record["nl_query"]
                    fout.write(json.dumps(record))
                    fout.write('\n')
                
    print("change query from kw to nl ok!")
        

def gen_nl_to_cnl_t5_inference_input_data(args):
    # ensure the query to be nl-query and formatted
    qid2query = pload("qid2query.pkl")
    with open(args.kw_to_nl_output_file, "r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = int(line["qid"])
            nl_query = line["nl_query"]
            qid2query[qid] = nl_query
    for qid in qid2query:
        qid2query[qid] = format_nl_query(qid2query[qid])
    
    qid2nlquery = qid2query
    pstore(qid2nlquery, "qid2nlquery.pkl")

    # transform nl to cnl
    with open(args.nl_to_cnl_inference_t5_input_file, "w") as fout, open(args.wss_meta_graph_file, "r") as fin:
        for line in tqdm(fin):
            line = json.loads(line)
            session_id = line["session_id"]
            turns = line["turns"]
            for turn_idx, turn in enumerate(turns):
                qid = turn["qid"]
                cur_query = qid2query[qid]
                IT_qid_list = turn["IT_qid_list"]
                RI_qid_sent_list = turn["RI_qid_sent_list"]
                
                for it_idx, it_qid in enumerate(IT_qid_list):
                    record = {}
                    record["sample_id"] = "{}_{}_{}_{}".format(session_id, turn_idx, "IT", it_idx)
                    record['oracle_query'] = qid2query[it_qid]
                    record["context"] = [cur_query]
                    fout.write(json.dumps(record))
                    fout.write('\n')
                
                for ri_idx, ri_item in enumerate(RI_qid_sent_list):
                    record = {}
                    record["sample_id"] = "{}_{}_{}_{}".format(session_id, turn_idx, "RI", ri_idx)
                    record['oracle_query'] = qid2query[ri_item[0]]
                    record['context'] = [ri_item[1], cur_query]
                    fout.write(json.dumps(record))
                    fout.write('\n')
    
    print("gen_nl_to_cnl_t5_inference_input_data ok!")


# process session: add nl to cnl to build the final session data file that we can perform random walk on.
def process_session_nl_to_cnl(args):
    qid2nlquery = pload("qid2nlquery.pkl")

    with open(args.wss_meta_graph_file) as fin:
        data = fin.readlines()
    sid2metagraph = {}
    # first set to nl-query
    for line in tqdm(data):
        line = json.loads(line)
        session_id = line["session_id"]
        turns = line["turns"]
        for i in range(len(turns)):
            turns[i]["query"] = qid2nlquery[turns[i]["qid"]]

            new_IT_qid_list = []
            for qid in turns[i]["IT_qid_list"]:
                new_IT_qid_list.append([qid, qid2nlquery[qid]])
            turns[i]["IT_qid_list"] = new_IT_qid_list

            new_RI_qid_sent_list = []
            for qid_sent in turns[i]["RI_qid_sent_list"]:
                new_RI_qid_sent_list.append([qid_sent[0], qid2nlquery[qid_sent[0]], qid_sent[1]])
            turns[i]["RI_qid_sent_list"] = new_RI_qid_sent_list
            
            if turns[i]["OI_qid"] != -1:
                turns[i]["OI_qid"] = (turns[i]["OI_qid"], qid2nlquery[turns[i]["OI_qid"]])
            
        sid2metagraph[session_id] = turns

    del data
    gc.collect()

    # replace to cnl-query
    with open(args.nl_to_cnl_inference_t5_output_file, "r") as fin:
        for line in tqdm(fin):
            try:
                line = json.loads(line)
            except:
                continue
            session_id, turn_idx, rel_type, idx = line["sample_id"].split("_")
            turn_idx = int(turn_idx)
            idx = int(idx)
            if rel_type == "IT":
                qid = sid2metagraph[session_id][turn_idx]["IT_qid_list"][idx][0]
                sid2metagraph[session_id][turn_idx]["IT_qid_list"][idx] = [qid, line["t5_cnl_query"]]
            elif rel_type == "RI":
                qid_query_sent = sid2metagraph[session_id][turn_idx]["RI_qid_sent_list"][idx]
                sid2metagraph[session_id][turn_idx]["RI_qid_sent_list"][idx] = [qid_query_sent[0], line["t5_cnl_query"], qid_query_sent[2]]       
      

    with open(args.cnl_meta_graph_file, "w") as fout:
        for sid in sid2metagraph:
            record = {}
            record["session_id"] = sid
            record["turns"] = sid2metagraph[sid]
            fout.write(json.dumps(record))
            fout.write('\n')
            
    print("process_session_nl_to_cnl ok! get cnl meta graph data.")


# perform random walk on cnl meta session graph
def rw_on_meta_session_graph(args):
    total_turn_num = 0
    total_session_num = 0
    with open(args.cnl_meta_graph_file, "r") as fin, open(args.secosearch_train_file, "w") as fout:
        for line in tqdm(fin):
            line = json.loads(line)
            session_id = line["session_id"]
            turns = line["turns"]
            
            res_turn_queries = []
            res_turn_qids = []
            global_upper_bound_turn_num = 0
            for turn in turns:
                global_upper_bound_turn_num += len(turn["IT_qid_list"]) + int(len(turn["RI_qid_sent_list"]) > 0)
                global_upper_bound_turn_num += 1
            
            upper_bound_total_turn_num = random.randint(min(5, global_upper_bound_turn_num), min(10, global_upper_bound_turn_num))
            for turn in turns:
                cur_qid = turn["qid"]
                cur_query = turn["query"]
                
                # cur query
                res_turn_qids.append(cur_qid)
                res_turn_queries.append(cur_query)
                if len(res_turn_qids) >= upper_bound_total_turn_num:
                    break
                
                # rw on IT
                add_it = False
                if random.randint(0, 4) <= 3:
                    add_it = True
                if add_it and len(turn["IT_qid_list"]) > 0:
                    it_num = random.randint(1, min([len(turn["IT_qid_list"]), 3, upper_bound_total_turn_num - len(res_turn_qids)]))
                    it_turns = random.sample(turn["IT_qid_list"], it_num)
                    for it_turn in it_turns:
                        res_turn_qids.append(it_turn[0])    # qid
                        res_turn_queries.append(it_turn[1])

                if len(res_turn_qids) >= upper_bound_total_turn_num:
                    break
                
                # rw on RI
                if len(turn["RI_qid_sent_list"]) == 0:
                    continue
                add_ri = False
                if random.randint(0, 9) <= 6:
                    add_ri = True
                if add_ri:
                    ri_turn = random.sample(turn["RI_qid_sent_list"], 1)[0]
                   
                    res_turn_qids.append(ri_turn[0])    # qid
                    res_turn_queries.append(ri_turn[1])
            
                
                if len(res_turn_qids) >= upper_bound_total_turn_num:
                    break

            record = {}
            record["session_id"] = session_id
            record["queries"] = res_turn_queries
            record["qids"] = res_turn_qids
            
            total_turn_num += len(res_turn_qids)
            total_session_num += 1

            fout.write(json.dumps(record))
            fout.write('\n')
    
    print("random walk on cnl meta session graph ok!")
    print("total_turn_num = {}".format(total_turn_num))
    print("total_session_num = {}".format(total_session_num))


def trans_session_to_train_samples(args):
    qrels = pload("qrels.pkl")
    with open(args.passage_file, "r") as f:
        passages = f.readlines()
    n_passage = len(passages)    
    print("load passage ok")

    with open(args.secosearch_train_file) as fin, open(args.final_generated_train_file, "w") as fout:
        for line in tqdm(fin):
            line = json.loads(line)
            queries = line["queries"]
            qids = line["qids"]
            session_id = line["session_id"]

            context_queries = []
            last_response = ""
            for i, query in enumerate(queries):
                qid = qids[i]
                if qid not in qrels:
                    context_queries.append(query)
                    last_response = None
                    continue
                record = {}
                record["sample_id"] = session_id + "_" + str(i)
                record["query"] = query
                record["last_response"] = last_response
                record["context_queries"] = context_queries
                
                pos_passage = qrels[qid][1]
                record["pos_docs"] = [pos_passage]
                neg_docs = []
                for _ in range(2):
                    neg_pid = random.randint(1, n_passage - 1)
                    _, neg_passage = passages[neg_pid].strip().split('\t')
                    neg_docs.append(neg_passage)
                record["neg_docs"] = neg_docs

                fout.write(json.dumps(record) + "\n")

                context_queries.append(query)
                last_response = pos_passage

                
                

    print("trans_session_to_train_samples OK!")     
            


def check_graph(args):
    qid2nlquery = pload("qid2nlquery.pkl")
    with open(args.cnl_meta_graph_file) as f:
        for line in tqdm(f):
            line = json.loads(line)
            session_id = line["session_id"]
            for turn in line["turns"]:
                query = turn["query"]
                print("-----------------")
                print(query)
                print('\n')
                for item in turn["IT_qid_list"]:
                    qid = item[0]
                    ts_query = qid2nlquery[qid]
                    cnl_ts_query = item[1]
                    print(ts_query)
                    print(cnl_ts_query)
                    print("\n")
                print("-----------------")
      



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")
    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    logger.info(args)
    
    return args



    #  this_turn["query"] = this_query
    #     this_turn["qid"] = this_qid
    #     this_turn["IT_qid_list"] = [x[0] for x in IT_qid_list]
    #     this_turn["RI_qid_sent_list"] = [(x[0],x[1]) for x in RI_qid_sent_list]
    #     this_turn["OI_qid"] = OI_qid



if __name__ == '__main__':
    args = get_args()
    args.device = torch.device("cuda:0")
    # get_all_qrels(args)
    # query_formatting_and_extract_topic_words(args)    
    process_session(args)
    # process_session_kw_to_nl(args)
    # gen_nl_to_cnl_t5_inference_input_data(args)
    # process_session_nl_to_cnl(args)
    # rw_on_meta_session_graph(args)
    # trans_session_to_train_samples(args)

    # check_graph(args)