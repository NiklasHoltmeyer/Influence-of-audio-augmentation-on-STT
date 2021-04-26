# Adopted/copied from: https://github.com/maxidl/wav2vec2/blob/c7ae26d36bb09062cad08be8702d7d68d5444f01/prepare_dataset.py

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torchaudio
from audioengine.model.finetuning.wav2vec2.parquetdataset import ParquetDataset
from tqdm.auto import tqdm
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from audioengine.corpus.dataset import Dataset
from audioengine.model.finetuning.wav2vec2.argument_parser import argument_parser

model_args, data_args, training_args = argument_parser(sys.argv)

# data_args.preprocess_dataset_path
# data_args.dataset_path
# increasing number of threads for torchaudio resample
print(f'Using {data_args.preprocessing_num_workers} threads')
torch.set_num_threads(data_args.preprocessing_num_workers)

chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'
print("Chars to Ignore", chars_to_ignore_regex)
print("Workers", data_args.preprocessing_num_workers)

assert data_args.dataset_path
assert data_args.preprocess_dataset_path

def load_datasets(validation_split=0.2):
    dataset_path = data_args.preprocess_dataset_path
    if "common" in dataset_path.lower() or "cv" in dataset_path.lower():
        return Dataset("huggingface").CommonVoice(data_args.preprocess_dataset_path, validation_split=validation_split)
    if "voxforge" in dataset_path.lower() or "vf" in dataset_path.lower():
        return Dataset("huggingface").VoxForge(data_args.preprocess_dataset_path, validation_split=validation_split)

train_dataset, eval_dataset = load_datasets()

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
vocab_test = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

resampled_data_dir = Path(data_args.dataset_path)
resampled_data_dir.mkdir(exist_ok=True)
vocab_path = Path(str(resampled_data_dir) + '/vocab.json').resolve()

with open(vocab_path, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

if data_args.max_train_samples is not None:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))

if data_args.max_val_samples is not None:
    eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def load_resample_save(f):
    f = Path(f)
    new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
    if not new_path.exists():
        speech_array, sampling_rate = torchaudio.load(f)
        resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
        speech_array_resampled = resampler(speech_array)
        input_values = processor(speech_array_resampled, sampling_rate=16_000).input_values
        input_values = torch.from_numpy(input_values).float().flatten()
        torch.save(input_values, new_path)
    return str(new_path)


print('load resample save')
new_train_paths = [load_resample_save(f) for f in tqdm(train_dataset['path'], miniters=100, desc='train')]
new_eval_paths = [load_resample_save(f) for f in tqdm(eval_dataset['path'], miniters=100, desc='eval')]

# update paths and sampling rate
train_dataset = train_dataset.map(
    lambda x: {'path': new_train_paths, 'sampling_rate': [16_000] * len(train_dataset), 'target_text': x['sentence']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    lambda x: {'path': new_eval_paths, 'sampling_rate': [16_000] * len(eval_dataset), 'target_text': x['sentence']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=eval_dataset.column_names,
)


# tokenize targets
def tokenize_targets(batch):
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


print('preparing dataset: train')

train_dataset = train_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in train_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)
print('preparing dataset: eval')

eval_dataset = eval_dataset.map(
    tokenize_targets,
    remove_columns=[col for col in eval_dataset.column_names if col != 'path'],
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)

pq.write_table(train_dataset.data, f'{resampled_data_dir}/{data_args.dataset_config_name}.train.parquet')
pq.write_table(eval_dataset.data, f'{resampled_data_dir}/{data_args.dataset_config_name}.eval.parquet')
print(f"Saved Pq`s to: {resampled_data_dir}")

print("Prepare: input_seq_lengths")
ds = ParquetDataset(data_args, split="train")
#def __init__(self, data_args, split='train'):

# save processor for training
print("Saving Processor")
processor.save_pretrained(training_args.output_dir)
