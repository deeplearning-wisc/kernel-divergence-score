
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd


def load_data(args, tokenizer, model_name, split=None):

    seen_dataset = pd.DataFrame(load_dataset('pratyushmaini/llm_dataset_inference', args.data, split='train', cache_dir=args.data_dir))
    unseen_dataset = pd.DataFrame(load_dataset('pratyushmaini/llm_dataset_inference', args.data, split='val', cache_dir=args.data_dir))
    
    seen_dataset['input'] = seen_dataset['text']
    unseen_dataset['input'] = unseen_dataset['text']

    seen_dataset['label'] = [1] * seen_dataset.shape[0]
    unseen_dataset['label'] = [0] * unseen_dataset.shape[0]

    seen_dataset = seen_dataset.to_dict('records')
    unseen_dataset = unseen_dataset.to_dict('records')
    contaminated_dataset = seen_dataset + unseen_dataset
    
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset