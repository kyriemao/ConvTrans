from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import os
from os import path
from os.path import join as oj
import os
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from utils import get_args, check_dir_exist_or_build, set_seed
from data_structure import Kw_to_Nl_with_T5_Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def save_model(args, model, tokenizer, save_model_order, epoch, step):
    output_dir = oj(args.model_output_path, 'model-{}-epoch-{}'.format(save_model_order, epoch))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))



def train_kw_to_nl_t5(args, log_writer):
    # model
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # training data and optimizer
    train_dataset = Kw_to_Nl_with_T5_Dataset(args, tokenizer, args.train_file_path)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=True,
                                  batch_size=args.batch_size, 
                                  collate_fn=train_dataset.get_collate_fn(mode="train"))

    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))    


    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_training_steps)

    # saving/log prepare
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
            model.train()
            
            bt_input_ids, bt_attention_mask, bt_labels = (batch["bt_input_ids"], batch["bt_attention_mask"], batch["bt_labels"])
            bt_input_ids = bt_input_ids.to(args.device)
            bt_attention_mask = bt_attention_mask.to(args.device)
            bt_labels = bt_labels.to(args.device)
        
            loss = model(input_ids=bt_input_ids, 
                         attention_mask=bt_attention_mask, 
                         labels=bt_labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
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
            log_writer.add_scalar("train_kw_to_nl_t5_loss", loss, global_step)
            global_step += 1    # avoid saving the model of the first step.
            
            # save model finally
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, tokenizer, save_model_order, epoch, global_step)
                save_model_order += 1

    logger.info("Training finish!")





if __name__ == '__main__':
    # prepare arguments
    args = get_args()
    set_seed(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(args.model_output_path) and os.listdir(
        args.model_output_path) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.model_output_path))

    check_dir_exist_or_build([args.model_output_path, args.log_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    

    log_writer = SummaryWriter(log_dir = args.log_dir_path)
    train_kw_to_nl_t5(args, log_writer)
    log_writer.close()