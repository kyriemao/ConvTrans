
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import toml
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import set_seed, top_p_filtering



def to_list(tensor):
    return tensor.detach().cpu().tolist()


class InferenceModel:

    def __init__(self, args):
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
        # special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = model_class.from_pretrained(args.model_path)
        # self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.model.to(args.device)
        self.model.eval()

        self.device = args.device
        self.length = args.max_output_seq_length
        if self.model.config.max_position_embeddings < args.max_output_seq_length:
            self.length = self.model.config.max_position_embeddings # No generation bigger than model size 
        self.temperature = args.temperature
        self.top_p = args.top_p

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

    def get_input_seq(self, context_queries, query):
        inputs = []
        for sent in context_queries:
            inputs.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)))
            inputs.append(self.tokenizer.sep_token_id)
        inputs.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query)))
        inputs.append(self.tokenizer.bos_token_id)
        return inputs

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, context_queries, query):
        context_queries = []
        input_ids = self.get_input_seq(context_queries, query)
        input_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # past_key_values = None
        with torch.no_grad():
            for step in range(self.length):
                inputs = {'input_ids': input_ids}
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[:, -1, :] / (self.temperature if self.temperature > 0 else 1.)
                # past_key_values = outputs.past_key_values
                filtered_logits = top_p_filtering(next_token_logits, top_p=self.top_p)
                if self.temperature == 0: # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                new_token = to_list(next_token)
                if self.tokenizer.decode(new_token[0]).strip() == "<EOS>":
                    break
                input_ids = torch.cat((input_ids, next_token), dim=1)

        pred_ids = to_list(input_ids[0, input_length:])
        pred_text = self.tokenizer.decode(pred_ids, clean_up_tokenization_spaces=True)
        pred_text = self.remove_special_tokens(pred_text)
        
        return pred_text 


def gpt_inference(args):
    inference_model = InferenceModel(args)
    with open(args.test_file_path , 'r', encoding="utf-8") as fin, open(args.output_file_path, 'w') as fout:
        for line in tqdm(fin, desc="Predict"):
            record = json.loads(line)
            prediction = inference_model.predict(record["context_queries"], record["query"])
            new_record = {}
            new_record["sample_id"] = record["sample_id"]
            new_record["query"] = record["query"]
            new_record['output'] = prediction
            fout.write(json.dumps(new_record) + '\n')



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
    # device = torch.device("cpu")
    args.device = device

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args





if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    gpt_inference(args)