
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import os
import json
import argparse
import toml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import set_seed, format_nl_query
from data_structure import Nl_to_Cnl_with_T5_Dataset


def inference_nl_to_cnl_t5(args):
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.add_tokens(["<CTX>", "<CUR>", "<SEP>"])
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    if args.n_gpu > 1:
        # query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))
        model = DDP(model, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    dist.barrier()

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    test_dataset = Nl_to_Cnl_with_T5_Dataset(args, tokenizer, args.test_file_path)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    ddp_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, 
                                  sampler=ddp_sampler,
                                  batch_size=args.batch_size, 
                                  collate_fn=test_dataset.get_collate_fn(mode="test"))
    
    # begin to inference
    with open(args.output_file_path, "a+") as f:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader, desc="Step", disable=args.disable_tqdm):
                bt_input_ids = batch["bt_input_ids"].to(args.device)
                bt_attention_mask = batch["bt_attention_mask"].to(args.device)
                output_seqs = model.module.generate(input_ids=bt_input_ids, 
                                             attention_mask=bt_attention_mask, 
                                             do_sample=False,
                                             max_length=args.max_nl_to_cnl_output_seq_length)
                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
                for i in range(len(outputs)):
                    record = {}
                    record["sample_id"] = batch["bt_sample_id"][i]
                    record["oracle_query"] = batch["bt_oracle_query"][i]
                    record["t5_cnl_query"] = format_nl_query(outputs[i])
                    record["context"] = batch["bt_context"][i]
                    f.write(json.dumps(record) + '\n') 

    
    logger.info("Nl to Cnl: Inference finsh!")
    

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

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    inference_nl_to_cnl_t5(args)