from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd


def load_data(args, tokenizer, model_name, split):

    dataset = load_dataset('truthfulqa/truthful_qa', 'generation', cache_dir=args.data_dir)['validation']
    dataset = pd.DataFrame(dataset)
    
    return format_ds(args, tokenizer, model_name, dataset)