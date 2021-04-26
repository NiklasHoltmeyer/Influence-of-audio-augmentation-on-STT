import os

from transformers import (
    HfArgumentParser,
    TrainingArguments, )

from audioengine.model.finetuning.wav2vec2.argument_classes import ModelArguments, DataTrainingArguments


def argument_parser(argv):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(argv) == 2 and argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args