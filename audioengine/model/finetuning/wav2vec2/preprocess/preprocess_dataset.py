# Adopted/copied from: https://github.com/maxidl/wav2vec2/blob/c7ae26d36bb09062cad08be8702d7d68d5444f01/prepare_dataset.py
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from transformers import (
    Wav2Vec2Processor, Wav2Vec2CTCTokenizer,
)

from audioengine.corpus.dataset import Dataset
from audioengine.corpus.util.text import save_settings
from audioengine.model.finetuning.wav2vec2.helper.argument_parser import argument_parser
from audioengine.model.finetuning.wav2vec2.helper.parquetdataset import ParquetDataset
from audioengine.model.finetuning.wav2vec2.preprocess.preprocess_data import load_resample_save, \
    remove_special_characters
from audioengine.model.finetuning.wav2vec2.preprocess.preprocess_dataset_settings import preprocess_settings
from audioengine.model.finetuning.wav2vec2.preprocess.vocab import build_vocab

model_args, data_args, training_args = argument_parser(sys.argv)

# data_args.preprocess_dataset_path
# data_args.dataset_path
# increasing number of threads for torchaudio resample
print(f'Using {data_args.preprocessing_num_workers} threads')
torch.set_num_threads(data_args.preprocessing_num_workers)

mappings = {
    'facebook/wav2vec2-large-xlsr-53-german': '[\,\?\.\!\-\;\:\"]',
    'maxidl/wav2vec2-large-xlsr-german': '[\,\?\.\!\-\;\:\"\“]',
    'marcel/wav2vec2-large-xlsr-53-german': '[\,\?\.\!\-\;\:\"\“\%\”\�\カ\æ\無\ན\カ\臣\ѹ\…\«\»\ð\ı\„\幺\א\ב\比\ш\ע\)\ứ\в\œ\ч\+\—\ш\‚\נ\м\ń\乡\$\=\ש\ф\支\(\°\и\к\̇]',
    'flozi00/wav2vec-xlsr-german': '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]',
    "marcel/wav2vec2-large-xlsr-german-demo": '[\,\?\.\!\-\;\:\"\“\%\”\�\カ\æ\無\ན\カ\臣\ѹ\…\«\»\ð\ı\„\幺\א\ב\比\ш\ע\)\ứ\в\œ\ч\+\—\ш\‚\נ\м\ń\乡\$\=\ש\ф\支\(\°\и\к\̇]',
    'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German': '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]',

    "facebook/wav2vec2-large-xlsr-53": '[\,\?\.\!\-\;\:\‘\”\�\']'  # <- change from original
}
# facebook/wav2vec2-large-xlsr-53

print("model_args.model_name_or_path", model_args.model_name_or_path)

chars_to_ignore_regex = mappings.get(model_args.model_name_or_path, None)
chars_to_ignore_regex = chars_to_ignore_regex if not None else f'[{"".join(data_args.chars_to_ignore)}]'

if model_args.processor_create_skip:
    chars_to_ignore_regex = mappings["facebook/wav2vec2-large-xlsr-53"]

target_sample_rate = 16_000
assert data_args.dataset_path, "Please set Flag dataset_path"

resampled_data_dir = Path(data_args.dataset_path)
resampled_data_dir.mkdir(exist_ok=True)

#
preprocess_settings = preprocess_settings()
(train_dataset, train_info), (eval_dataset, eval_info) = Dataset("huggingface").from_settings(preprocess_settings)

ds_settings_path = f"{resampled_data_dir}/dataset_split.json"
save_settings(ds_settings_path, preprocess_settings, [("train_info", train_info),
                                                      ("eval_info", eval_info)])
train_dataset = train_dataset.map(remove_special_characters(chars_to_ignore_regex), remove_columns=["sentence"],
                                  keep_in_memory=True,
                                  num_proc=data_args.preprocessing_num_workers)

eval_dataset = eval_dataset.map(remove_special_characters(chars_to_ignore_regex), remove_columns=["sentence"],
                                keep_in_memory=True,
                                num_proc=data_args.preprocessing_num_workers)

if not model_args.processor_create_skip:
    vocab_path = Path(str(resampled_data_dir) + '/vocab.json').resolve()
    vocab_dict = build_vocab(train_dataset, eval_dataset)

    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=target_sample_rate, padding_value=0.0, do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
else:
    vocab_path = model_args.model_name_or_path
    processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)

if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))

if data_args.max_val_samples is not None and data_args.max_val_samples > 0:
    eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

# processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
load_resample_save_fn = load_resample_save(resampled_data_dir, processor, target_sample_rate)
new_train_paths = [load_resample_save_fn(f)
                   for f in tqdm(train_dataset['path'], miniters=100, desc='resample (train)')]
new_eval_paths = [load_resample_save_fn(f)
                  for f in tqdm(eval_dataset['path'], miniters=100, desc='resample (eval)')]

# new_train_paths = [x for x in new_train_paths if x is not None]
# new_eval_paths = [x for x in new_train_paths if x is not None]


# update paths and sampling rate
train_dataset = train_dataset.map(
    lambda x: {'path': new_train_paths, 'sampling_rate': [target_sample_rate] * len(train_dataset),
               'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    lambda x: {'path': new_eval_paths, 'sampling_rate': [target_sample_rate] * len(eval_dataset),
               'target_text': x['text']},
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

processor.save_pretrained(training_args.output_dir)

pq.write_table(train_dataset.data.table, f'{resampled_data_dir}/{data_args.dataset_config_name}.train.parquet')
pq.write_table(eval_dataset.data.table, f'{resampled_data_dir}/{data_args.dataset_config_name}.eval.parquet')
print(f"Saved Pq`s to: {resampled_data_dir}")

print("Prepare: input_seq_lengths")
ds = ParquetDataset(data_args, split="train")

print("train_info:", train_info)
print("eval_info:", eval_info)

print("model_name_or_path", model_args.model_name_or_path)
print("vocab_path", vocab_path)
print("vocab_trained", not model_args.processor_create_skip)
print("Processor Path", training_args.output_dir)
