
import os, copy, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import argparse, sys
import random, json, pickle, re, collections
from tqdm import tqdm
from datetime import datetime

from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from tqdm import tqdm
import datasets
from datasets import Dataset, concatenate_datasets
from transformers import (
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from data.data_utils import *
from model.model_utils import *
from profiler import Profiler


os.environ['TOKENIZERS_PARALLELISM'] = "True"

def train_model(args, model, dataset):

    # Format Dataset
    _dataset = Dataset.from_dict({"text": dataset["input"]})
    def tokenize_function(examples):
        tokens = model.tokenizer(examples["text"], padding=True, truncation=True)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    
    # caching for large models is broken due to the fingerprint calculation by hf datasets
    # we need to force disable caching
    was_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    _dataset = _dataset.map(tokenize_function, batched=True, remove_columns=["text"], new_fingerprint="DO_NOT_ENABLE_CACHING", cache_file_name=None)
    if was_enabled: datasets.enable_caching()

    # for phi models, we need a different targeted modules list due to
    # the layer modules being named differently. (fun fact the phi approach is more effecient)
    targeted_modules_exceptions = {
        "phi3-small": ["query_key_value"],
        "phi3-medium": ["qkv_proj"],
        "internlm": ["wqkv"],
        "bloomz": ["query_key_value"],
    }.get(args.model, ["q_proj", "v_proj"])

    # LoRA arguments
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Adjust to your model's task, e.g., SEQ_2_SEQ_LM for seq-to-seq
        inference_mode=False,
        r=args.lora_dim,                           # Rank of the update matrices
        lora_alpha=args.lora_alpha,                # Scaling factor
        lora_dropout=0.1,                          # Dropout rate for LoRA
        target_modules=targeted_modules_exceptions,
    )
    _model = get_peft_model(model.huggingface_model, lora_config)

    # Training arguments
    if not os.path.exists('out/sft_output/'):
        os.makedirs('out/sft_output/')
    
    if args.sgd :
        training_args = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            optim="sgd",  
            weight_decay=0.0,
            max_grad_norm=None,
            logging_dir="out/sft_output/",
            report_to=[],
            run_name=args.exp_name
        )
    else :
        training_args = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            optim="sgd",  
            weight_decay=0.0,
            max_grad_norm=None,
            logging_dir="out/sft_output/",
            report_to=[],
            run_name=args.exp_name,
            max_steps=args.epochs,  # Force only one optimization step per epoch
            gradient_accumulation_steps=int(np.ceil(len(_dataset) // (args.batch_size))),
        )

    optimizer = torch.optim.SGD(
        _model.parameters(),
        lr=training_args.learning_rate
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset,
        eval_dataset=_dataset,
        tokenizer=model.tokenizer,
        optimizers=(optimizer, None),
        data_collator=DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()

    model.huggingface_model = _model
    return model



def main(args):

    # 1. Load Dataset & Model
    model = load_model(args)

    full_dataset = load_data(args, model.tokenizer, args.model, split=args.split) # in-data is labeled 1, else 0
    in_dataset, out_dataset, in_labels, out_labels = get_data_subsets(args, full_dataset)
    dataset = concatenate_datasets([in_dataset, out_dataset])
    labels = in_labels + out_labels

    # 2. Setup Profiler
    profiler = Profiler(args)

    # 3. Setup Pre SFT
    pre_embs = profiler.get_embeddings(args, model, dataset)
    
    # 4. Train Model
    model = train_model(args, model, dataset)

    # 5. Setup Post SFT
    post_embs = profiler.get_embeddings(args, model, dataset)

    # 6. Profiling
    profiler.profile(args, model, dataset, pre_embs, post_embs)


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--out_dir', type=str, default="")

    # model
    parser.add_argument('--model', type=str, default='llama3')
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dim', type=int, default=8)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--inference_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--sgd', action='store_true')

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='beavertails')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--target_num', type=int, default=1000)
    parser.add_argument('--contamination', type=float, default=-1)
    parser.add_argument('--cpu_profiler', action='store_true')

    parser.add_argument('--perturbation', type=float, default=0.05)
    parser.add_argument('--answer_level_shuffling', action='store_true')
    parser.add_argument('--synonym_replacement', action='store_true')
    parser.add_argument('--random_deletion', action='store_true')
    parser.add_argument('--word_level_shuffling', action='store_true')


    # KDS Profiler
    parser.add_argument('--gamma', type=float)



    return parser.parse_args()


if __name__ == '__main__' :

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    with open('token','r') as f :
        token = f.read()
    args.token = token
    
    main(args)
