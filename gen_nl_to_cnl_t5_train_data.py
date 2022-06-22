from concurrent.futures import process
from IPython import embed
import json
from utils import get_args
from tqdm import tqdm
import re
from utils import BIG_STOP_WORDS

# gen nl_to_cnl_t5_train_data from CAsT-19,20,21

def query_formatting(q):
    q = re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "", q)

    q = q.lower()

    q = q.split(" ")
    for i in range(len(q)):
        q[i] = q[i].replace(" ", "")
    q = " ".join(q)
    q = q.rstrip().lstrip()
    
    return q

def get_last_response_sent(query, oracle_query, passage):
    query = query_formatting(query)
    oracle_query = query_formatting(oracle_query)
    query_tokens = set(query.split(" "))
    oracle_query_tokens = set(oracle_query.split(" "))
    new_tokens = oracle_query_tokens - query_tokens - set(BIG_STOP_WORDS)
    threshold = len(new_tokens) / 2
    res_sents = []
    sents = re.split('[.?!]', passage)
    for sent in sents:
        sent = sent.rstrip().lstrip()
        if sent == "":
            continue
        pro_sent = query_formatting(sent)    
        sent_tokens = set(pro_sent.split(" "))
        intersect_num = len(new_tokens & sent_tokens)
        if intersect_num > threshold:
            res_sents.append(sent)
                

    if len(res_sents) == 0:
        return passage
    if len(res_sents) == 1:
        return res_sents[0] + "."
    return ". ".join(res_sents)


def get_sid2depenid(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()

    sid2depenid = {}
    is_session = True
    session_id = -1
    for line in data:
        line = line.strip()
        line = line.replace(" ", "")
        if is_session:
            session_id = int(line)
            is_session = False
        elif line == "":
            is_session = True
        else:
            line = line.split(",")
            turn_id = int(line[0])
            sample_id = "{}_{}".format(session_id, turn_id)
            for depen_id in line[1:]:
                depen_sample_id = "{}_{}".format(session_id, depen_id)
                if sample_id not in sid2depenid:
                    sid2depenid[sample_id] = []
                sid2depenid[sample_id].append(depen_sample_id)
    for sid in sid2depenid:
        sid2depenid[sid] = sorted(sid2depenid[sid])
    return sid2depenid


def process_cast19(args):
    with open(args.cast19_test_file, "r") as f:
        data = f.readlines()
    sid2query = {}
    for line in data:
        line = json.loads(line)
        sample_id = line["sample_id"]
        sid2query[sample_id] = [line["query"], line["oracle_query"]]
    sid2depenid = get_sid2depenid(args.cast19_turn_depen_file)

    with open(args.cast19_output_file, "w") as f:
        for sample_id in sid2depenid:
            record = {}
            record["sample_id"] = sample_id
            record['query'] = sid2query[sample_id][0]
            record['oracle_query'] = sid2query[sample_id][1]
            
            # context
            record['context_queries'] = []
            record['context_oracle_queries'] = []
            # record['cnl_t5_context'] = []
            for depen_id in sid2depenid[sample_id]:
                record['context_queries'].append(sid2query[depen_id][0])
                record['context_oracle_queries'].append(sid2query[depen_id][1])
            
            record['context'] = record['context_oracle_queries']
            
            f.write(json.dumps(record))
            f.write('\n')

    print('gen cast19 nl to cnl train data ok!')



def process_cast20(args):
    with open(args.cast20_raw_annotated_file, "r") as f:
        data = json.load(f)
    
    sid2query = {}
    response_depen_turn_set = set()
    for session in data:
        session_id = session["number"]
        for turn in session["turn"]:
            turn_id = turn["number"]
            sample_id = "{}_{}".format(session_id, turn_id)
            query = turn["raw_utterance"]
            try:
                oracle_query = turn["manual_rewritten_utterance"]
            except:
                oracle_query = query
            sid2query[sample_id] = [query, oracle_query]
            if "result_turn_dependence" in turn:
                response_depen_turn_set.add(sample_id)

    sid2depenid = get_sid2depenid(args.cast20_turn_depen_file)
    
    sid2lastresp = {}
    with open(args.cast20_test_file, "r") as f:
        for line in f:
            line = json.loads(line)
            sid2lastresp[line["sample_id"]] = line["last_response"]

    # sid2query, sid2depenid, response_depen_turn_set, sid2lastresp
    with open(args.cast20_output_file, "w") as f:
        for sample_id in sid2depenid:
            record = {}
            record["sample_id"] = sample_id
            record['query'] = sid2query[sample_id][0]
            record['oracle_query'] = sid2query[sample_id][1]
            
            # context
            record['context_queries'] = []
            record['context_oracle_queries'] = []
            record['context'] = []
            for depen_id in sid2depenid[sample_id]:
                record['context_queries'].append(sid2query[depen_id][0])
                record['context_oracle_queries'].append(sid2query[depen_id][1])
            if sample_id in response_depen_turn_set:
                # record['cnl_t5_context'].append(sid2lastresp[sample_id])
                last_response = sid2lastresp[sample_id]
                last_sent = get_last_response_sent(record['query'], record['oracle_query'], last_response)
                record['context'].append(last_sent)
                conv_idx, turn_idx = sample_id.split("_")
                turn_idx = int(turn_idx)
                last_sample_id = "{}_{}".format(conv_idx, turn_idx - 1)
                record['context'].append(sid2query[last_sample_id][1])
            else:
                record['context'] = record['context_oracle_queries']
            
            f.write(json.dumps(record))
            f.write('\n')

    print('gen cast20 nl to cnl train data ok!')

                                 





if __name__ == "__main__":
    args = get_args()
    process_cast19(args)
    process_cast20(args)