import os
import time
from argparse import ArgumentTypeError
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.metrics.wer import Jiwer
from audioengine.model.pretrained.wav2vec2 import wav2vec2


def validate_model(model_name, based_on=None):
    w2c = wav2vec2(model_name, based_on=based_on)
    transform = w2c.transformation()

    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": False,
        "validation_split": None,
        "type": "test",
        "min_target_length": 2,
    }

    ds_settings = {"val_settings": [cv_test_full], "train_settings": None, "transform": transform}
    (_, _), (ds, ds_info) = Dataset("torch").from_settings(ds_settings)

    core_count = os.cpu_count()  # batch-size -> 20
    dataloader = DataLoader(ds, batch_size=16, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    wer = Jiwer()
    sentence_stacked = []
    transcriptions_stacked = []
    start_time = time.time()
    for idx, (speeches, sentences) in enumerate(tqdm(dataloader, desc=model_name)):
        transcriptions = w2c.predict(speeches)
        transcriptions_stacked.extend(transcriptions)
        sentence_stacked.extend(sentences)

        if idx % 97 == 0:  # 97 71
            wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
            sentence_stacked, transcriptions_stacked = [], []

    return wer.to_tsv(prefix=model_name, suffix=str(time.time() - start_time)).replace(".", ",")


def in_list(_list, exception_text):
    def __call__(item):
        if not item in _list:
            raise ArgumentTypeError(exception_text + item)
        return item

    return __call__


nh_proc = "unknownTransformer/wav2vec2-large-xlsr-german"

jobs = [
    '/share/modelle/tts/models/cv-tts-nofilter-nh',
]

results = []
failed = []


##/share/modelle/vf_cv_small/results.tsv
##/share/modelle/vf_cv_small/failed.tsv

def already_run(model_name, file_path):
    with open(file_path, "r+") as f:
        for line in f.readlines():
            line_model = line.split("\t")[0]
            if model_name in line_model:
                return True
    return False
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
result_tsv = validate_model(model_name, based_on=model_name)
print(result_tsv)

#supported_models = ['facebook/wav2vec2-large-xlsr-53-german',
#                    'maxidl/wav2vec2-large-xlsr-german',
#                    'marcel/wav2vec2-large-xlsr-53-german',
#                    'flozi00/wav2vec-xlsr-german',
#                    'marcel/wav2vec2-large-xlsr-german-demo',
#                    'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German']
#parser_supported_models_str = ["\t" + model for model in supported_models]
#parser_supported_models_str = "Supported Models: \r\n" + "\r\n".join(parser_supported_models_str)

#parser = argparse.ArgumentParser(description="Evaluate Wav2Vec", formatter_class=RawTextHelpFormatter)
#parser.add_argument('--model_name', '-m', required=True,
#                   help=parser_supported_models_str, type=in_list(supported_models, "\r\nInvalid Model Name: "))

#args = parser.parse_args()
#model_name = args.model_name
#model_name = "flozi00/wav2vec-xlsr-german"
#for model_name in supported_models:
#    try:
#       print(validate_model(model_name))
#   except Exception as e:
#       error = "\t".join([model_name, "error", str(e)])
#       print(error)
