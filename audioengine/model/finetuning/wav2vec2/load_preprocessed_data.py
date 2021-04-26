from pathlib import Path

from audioengine.model.finetuning.wav2vec2.argument_parser import argument_parser
from audioengine.model.finetuning.wav2vec2.parquetdataset import ParquetDataset

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
    import sys
    model_args, data_args, training_args = argument_parser(sys.argv)
