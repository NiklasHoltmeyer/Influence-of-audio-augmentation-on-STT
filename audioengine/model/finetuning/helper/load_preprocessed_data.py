from pathlib import Path

from audioengine.model.finetuning.helper.ParquetDataset import ParquetDataset


def load_datasets(data_args):
    train_dataset = ParquetDataset(data_args, split='train')
    eval_dataset = ParquetDataset(data_args, split='eval')
    return train_dataset, eval_dataset

def get_vocab_path(data_args):
    resampled_data_dir = Path(data_args.dataset_path)
    resampled_data_dir.mkdir(exist_ok=True)
    vocab_path = Path(str(resampled_data_dir) + '/vocab.json')
    assert vocab_path.exists()
    return vocab_path.resolve()

def get_preprocessor_path(training_args):
    return training_args.output_dir


if __name__ == "__main__":
    import os
    import sys

    from transformers import (
        HfArgumentParser,
        TrainingArguments,
    )

    from argument_classes import ModelArguments, DataTrainingArguments
    from audioengine.model.finetuning.helper.ParquetDataset import ParquetDataset

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
