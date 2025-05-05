
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd



def load_data(args, tokenizer, model_name, split=None):

    dataset = pd.DataFrame(load_dataset('swj0419/BookMIA', split=split, cache_dir=args.data_dir))
    dataset = dataset[[True if len(x) < 2930 else False for x in dataset['snippet']]]
    dataset['input'] = dataset['snippet']

    seen_dataset = dataset[dataset.label == 1].to_dict('records')
    unseen_dataset = dataset[dataset.label == 0].to_dict('records')
    contaminated_dataset = seen_dataset + unseen_dataset
    
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset