
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd


def load_data(args, tokenizer, model_name, split=None):
    dataset = pd.DataFrame(load_dataset('avduarte333/arXivTection', cache_dir=args.data_dir)[split]).sample(frac=1)
    dataset['input'] = dataset['Example_A']
    dataset['label'] = dataset['Label']

    seen_dataset = dataset[dataset.label == 1].to_dict('records')
    unseen_dataset = dataset[dataset.label == 0].to_dict('records')
    contaminated_dataset = seen_dataset + unseen_dataset

    dataset = Dataset.from_list(contaminated_dataset)
    return dataset