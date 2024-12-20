from datasets import load_dataset

from train.arguments import DataArguments


def load_train_data(dataArgs: DataArguments):
    seed = 42
    if dataArgs.load_local_dataset:
        datasets = load_dataset(dataArgs.train_data_format, data_files=dataArgs.train_data_path)
    else:
        datasets = load_dataset(dataArgs.train_data_path)
    if dataArgs.train_data_percentage < 1.0:
        datasets = datasets.shuffle(seed=seed).select(range(int(len(datasets) * dataArgs.train_data_percentage)))

    datasets = datasets['train'].train_test_split(test_size=dataArgs.test_datasets_percentage, shuffle=True, seed=seed)
    print("train data one sample like: ", datasets['train'][0], "train sample rows:", datasets['train'].num_rows)
    return datasets
