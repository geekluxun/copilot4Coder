import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk


def merge_hf_datasets_streaming_to_parquet_zip_parallel(
        datasets_info,
        output_path,
        records_per_shard=100000,
        max_workers=4
):
    def get_iterator(dataset_info):
        dataset = load_dataset(
            dataset_info['name'],
            dataset_info.get('config', None),
            split=dataset_info['split'],
            streaming=True
        )
        print(dataset)
        feature1, feature2 = dataset_info['features']
        for example in dataset:
            yield {
                'prompt': example[feature1],
                'completion': example[feature2]
            }

    def save_shard_as_parquet_zip(shard_data, shard_idx, output_path):
        try:
            start_time = time.time()
            process_id = os.getpid()
            print(
                f"[Shard {shard_idx}] Process {process_id} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

            shard_dataset = Dataset.from_dict({
                'prompt': [r['prompt'] for r in shard_data],
                'completion': [r['completion'] for r in shard_data]
            })

            parquet_filename = f'shard_{shard_idx}.parquet'
            parquet_path = os.path.join(output_path, parquet_filename)
            zip_filename = f'shard_{shard_idx}.zip'
            zip_path = os.path.join(output_path, zip_filename)

            shard_dataset.to_parquet(parquet_path)
            save_time = time.time()
            print(
                f"[Shard {shard_idx}] Process {process_id} saved Parquet at {parquet_path} in {save_time - start_time:.2f} seconds")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(parquet_path, arcname=parquet_filename)
            compress_time = time.time()
            print(
                f"[Shard {shard_idx}] Process {process_id} compressed to {zip_path} in {compress_time - save_time:.2f} seconds")

            os.remove(parquet_path)
            print(
                f"[Shard {shard_idx}] Process {process_id} removed temporary file {parquet_path} at {time.time() - compress_time:.2f} seconds")
        except Exception as e:
            print(f"[Shard {shard_idx}] Process {os.getpid()} encountered error: {e}")

    iterators = [get_iterator(info) for info in datasets_info]
    merged_iterator = chain(*iterators)
    os.makedirs(output_path, exist_ok=True)

    shard_idx = 1
    current_shard = []
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for record in merged_iterator:
            current_shard.append(record)
            if len(current_shard) >= records_per_shard:
                future = executor.submit(save_shard_as_parquet_zip, list(current_shard), shard_idx, output_path)
                futures.append(future)
                print(f"[Shard {shard_idx}] Submitted to executor at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                shard_idx += 1
                current_shard = []

        if current_shard:
            future = executor.submit(save_shard_as_parquet_zip, list(current_shard), shard_idx, output_path)
            futures.append(future)
            print(f"[Shard {shard_idx}] Submitted to executor at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in saving shard: {e}")

    print("所有分片已成功保存为压缩的 Parquet 文件。")


# 把多个数据集合并成一个数据集（子集的形式）
def combine_datasets(datasets_info, combined_dataset_path):
    combined_datasets = DatasetDict()
    for dataset_info in datasets_info:
        dataset = load_dataset(
            dataset_info['name'],
            dataset_info.get('config', None),
            split=dataset_info['split']
        )
        new_columns = ["prompt", "completion"]
        # 创建重命名映射字典
        rename_mapping = dict(zip(dataset_info['features'], new_columns))
        print(f"重命名映射：{rename_mapping}")
        dataset = dataset.rename_columns(rename_mapping)
        # 仅保留重命名后的两列
        dataset = dataset.select_columns(new_columns)
        config = dataset_info['name'].split("/")[-1]
        combined_datasets[config] = dataset
    combined_datasets.save_to_disk(combined_dataset_path, max_shard_size="200MB")


# 转成jsong格式
def covert_dataset_tojson(dataset_path, output_name):
    dataset = load_from_disk(dataset_path=dataset_path)
    print(dataset.num_rows)
    dataset.to_json(f"data/{output_name}")


if __name__ == "__main__":
    datasets_info = [
        {
            "name": "code-search-net/code_search_net",
            "split": "train",
            "features": ["func_documentation_string", "func_code_string"]
        },
        {
            "name": "HuggingFaceH4/CodeAlpaca_20K",
            "split": "train",
            "features": ["prompt", "completion"]
        },
    ]

    # case 1
    combine_datasets(datasets_info, "combined_dataset")

    # case 2
    # covert_dataset_tojson(
    #     dataset_path="/src/data/merged_dataset/code_search_net",
    #     output_name="code_search_net.jsonl")

    # case3
    # output_path = "merged_dataset"
    # records_per_shard = 100000  # 每个分片包含的记录数
    # max_workers = 10  # 根据您的机器性能调整进程数
    #
    # merge_hf_datasets_streaming_to_parquet_zip_parallel(
    #     datasets_info,
    #     output_path,
    #     records_per_shard,
    #     max_workers
    # )
