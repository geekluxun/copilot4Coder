from datasets import load_dataset

from train.arguments import MyDataArguments


def load_train_data(dataArgs: MyDataArguments):
    seed = 42
    if dataArgs.load_local_dataset:
        datasets = load_dataset(dataArgs.train_data_format, data_dir=dataArgs.train_data_path)
    else:
        datasets = load_dataset(dataArgs.train_data_path)

    datasets = datasets['train'].shuffle(seed=seed).select(range(int(len(datasets['train']) * dataArgs.train_data_percentage)))

    datasets = datasets.train_test_split(test_size=dataArgs.test_datasets_percentage, shuffle=True, seed=seed)
    print("train data one sample like: ", datasets['train'][0])
    print("train sample rows:", datasets['train'].num_rows)
    return datasets
