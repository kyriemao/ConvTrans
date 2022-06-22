
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import get_args, set_seed, format_nl_query
from data_structure import Kw_to_Nl_with_T5_Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def inference_kw_to_nl_t5(args):
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_path)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    test_dataset = Kw_to_Nl_with_T5_Dataset(args, tokenizer, args.test_file_path)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset, 
                                  shuffle=False,
                                  batch_size=args.batch_size, 
                                  collate_fn=test_dataset.get_collate_fn(mode="test"))
    
    # begin to inference
    with open(args.output_file_path, "w") as f:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader, desc="Step", disable=args.disable_tqdm):
                bt_input_ids = batch["bt_input_ids"].to(args.device)
                bt_attention_mask = batch["bt_attention_mask"].to(args.device)
                output_seqs = model.generate(input_ids=bt_input_ids, 
                                             attention_mask=bt_attention_mask, 
                                             do_sample=False,
                                             max_length=args.max_seq_length)
                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
                for i in range(len(outputs)):
                    record = {}
                    record["sample_id"] = batch["bt_sample_id"][i]
                    record["kw_query"] = batch["bt_kw_query"][i]
                    record["t5_oracle_query"] = format_nl_query(outputs[i])
                    f.write(json.dumps(record))
                    f.write('\n')
    
    logger.info("Kw to Nl: Inference finsh!")
    

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    inference_kw_to_nl_t5(args)