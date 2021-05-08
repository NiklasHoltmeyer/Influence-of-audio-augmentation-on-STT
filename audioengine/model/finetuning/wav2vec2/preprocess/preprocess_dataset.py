# Adopted/copied from: https://github.com/maxidl/wav2vec2/blob/c7ae26d36bb09062cad08be8702d7d68d5444f01/prepare_dataset.py

import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torchaudio

from audioengine.model.finetuning.wav2vec2.helper.parquetdataset import ParquetDataset
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
)

from audioengine.corpus.dataset import Dataset
from audioengine.model.finetuning.wav2vec2.helper.argument_parser import argument_parser
from audioengine.model.finetuning.wav2vec2.preprocess.preprocess_dataset_settings import preprocess_settings
from audioengine.corpus.util.text import save_settings

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
            'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German': '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
}

print("model_args.model_name_or_path", model_args.model_name_or_path)

chars_to_ignore_regex = mappings.get(model_args.model_name_or_path, None)
chars_to_ignore_regex = chars_to_ignore_regex if not None else f'[{"".join(data_args.chars_to_ignore)}]'


assert data_args.dataset_path, "Please set Flag dataset_path"
# assert data_args.preprocess_dataset_train_path, "Please set Flag preprocess_dataset_train_path"
# assert data_args.preprocess_dataset_eval_path, "Please set Flag preprocess_dataset_eval_path"

resampled_data_dir = Path(data_args.dataset_path)
resampled_data_dir.mkdir(exist_ok=True)

#
preprocess_settings = preprocess_settings()
(train_dataset, train_info), (eval_dataset, eval_info) = Dataset("huggingface").from_settings(preprocess_settings)

ds_settings_path = f"{resampled_data_dir}/dataset_split.json"
save_settings(ds_settings_path, preprocess_settings, [("train_info", train_info),
                                                      ("eval_info", eval_info)])


def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower() + " "
    return batch


train_dataset = train_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True,
                                  num_proc=data_args.preprocessing_num_workers)

eval_dataset = eval_dataset.map(remove_special_characters, remove_columns=["sentence"], keep_in_memory=True,
                                num_proc=data_args.preprocessing_num_workers)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# vocab_train = train_dataset.map(
#    extract_all_chars,
#    batched=True,
#    batch_size=-1,
#    keep_in_memory=True,
#    remove_columns=train_dataset.column_names,
# )
# vocab_test = eval_dataset.map(
#    extract_all_chars,
#    batched=True,
#    batch_size=-1,
#    keep_in_memory=True,
#    remove_columns=eval_dataset.column_names,
# )
#
# vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
# vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)

vocab_path = Path(str(resampled_data_dir) + '/vocab.json').resolve()

# with open(vocab_path, 'w') as vocab_file:
#    json.dump(vocab_dict, vocab_file)

if data_args.max_train_samples is not None:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))

if data_args.max_val_samples is not None:
    eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

# tokenizer = Wav2Vec2CTCTokenizer(
#    vocab_path,
#    unk_token="[UNK]",
#    pad_token="[PAD]",
#    word_delimiter_token="|",
# )
##feature_extractor = Wav2Vec2FeatureExtractor(
##    feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
# )
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
target_sample_rate = 16_000


def load_resample_save(f):
    f = Path(f)
    new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
    if not new_path.exists():
        speech_array, sampling_rate = torchaudio.load(f)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sample_rate)
        speech_array_resampled = resampler(speech_array)
        input_values = processor(speech_array_resampled, sampling_rate=target_sample_rate).input_values
        input_values = torch.from_numpy(input_values).float().flatten()

        torch.save(input_values, new_path)
    return str(new_path)


new_train_paths = [load_resample_save(f)
                  for f in tqdm(train_dataset['path'], miniters=100, desc='resample (train)')]
new_eval_paths = [load_resample_save(f)
                 for f in tqdm(eval_dataset['path'], miniters=100, desc='resample (eval)')]

# new_train_paths = [x for x in new_train_paths if x is not None]
# new_eval_paths = [x for x in new_train_paths if x is not None]


# update paths and sampling rate
train_dataset = train_dataset.map(
    lambda x: {'path': new_train_paths, 'sampling_rate': [16_000] * len(train_dataset), 'target_text': x['text']},
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    lambda x: {'path': new_eval_paths, 'sampling_rate': [16_000] * len(eval_dataset), 'target_text': x['text']},
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

pq.write_table(train_dataset.data.table, f'{resampled_data_dir}/{data_args.dataset_config_name}.train.parquet')
pq.write_table(eval_dataset.data.table, f'{resampled_data_dir}/{data_args.dataset_config_name}.eval.parquet')
print(f"Saved Pq`s to: {resampled_data_dir}")

print("Prepare: input_seq_lengths")
ds = ParquetDataset(data_args, split="train")

print("train_info:", train_info)
print("eval_info:", eval_info)

print("model_name_or_path", model_args.model_name_or_path)
# def __init__(self, data_args, split='train'):

# save processor for training
# print(f"Saving Processor to {training_args.output_dir}")
# processor.save_pretrained(training_args.output_dir)

