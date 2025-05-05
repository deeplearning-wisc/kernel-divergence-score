
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

from data.data_utils import *


def load_data(args, tokenizer, model_name, split=None):
    sets = []
    for subset in ['WikiMIA_length32','WikiMIA_length64','WikiMIA_length128','WikiMIA_length256']:
        sets.append(load_dataset('swj0419/WikiMIA', cache_dir=args.data_dir)[subset])
    dataset = pd.DataFrame(concatenate_datasets(sets)).sample(frac=1)

    seen_dataset = dataset[dataset.label == 1].to_dict('records')
    unseen_dataset = dataset[dataset.label == 0].to_dict('records')
    contaminated_dataset = seen_dataset + unseen_dataset
    
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset