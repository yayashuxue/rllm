from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    dataset = load_dataset("kylemontgomery/imo", split="train")

    def preprocess_fn(example):
        return {
            "id": example["id"],
            "question": example["problem"],
            "ground_truth": None,
            "data_source": "imo",
        }

    dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

    train_dataset = dataset.filter(lambda e: not e["id"].startswith("IMO-2025"))
    test_dataset = dataset.filter(lambda e: e["id"].startswith("IMO-2025"))

    print(train_dataset)
    print(test_dataset)

    train_dataset = DatasetRegistry.register_dataset("imo", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("imo", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
