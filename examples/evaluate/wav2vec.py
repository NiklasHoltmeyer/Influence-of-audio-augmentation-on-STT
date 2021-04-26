import argparse
from argparse import RawTextHelpFormatter, ArgumentTypeError
import torch
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.metrics.wer import Jiwer
from torchvision import transforms
from tqdm import tqdm
from audioengine.model.pretrained.wav2vec2 import wav2vec2
from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from torch.utils.data import DataLoader
import os
import time

from audioengine.transformations.backend.pytorch.texttransformations import ToUpper


def validate_model(model_name, based_on = None):
    w2c = wav2vec2(model_name, based_on=based_on)
    transformations = w2c.transformations()

    if model_name == "flozi00/wav2vec-xlsr-german":
        transformations[0] =ToUpper("sentence")

    transform = transforms.Compose(transformations)

    dataset = Dataset("torch").CommonVoice("/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de", shuffle=False,
                                           transform=transform, type="test", validation_split=None)
    core_count = os.cpu_count()

    dataloader = DataLoader(dataset, batch_size=16, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    wer = Jiwer()
    sentence_stacked = []
    transcriptions_stacked = []
    start_time = time.time()
    for idx, (speeches, sentences) in enumerate(tqdm(dataloader)):
        transcriptions = w2c.predict(speeches)
        transcriptions_stacked.extend(transcriptions)
        sentence_stacked.extend(sentences)

        if idx % 97 == 0: #97 71
            wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
            sentence_stacked, transcriptions_stacked = [], []

    return wer.to_tsv(prefix=model_name, suffix=str(time.time()-start_time)).replace(".", ",")

def in_list(_list, exception_text):
    def __call__(item):
        if not item in _list:
            raise ArgumentTypeError(exception_text + item)
        return item
    return __call__

based_on = "maxidl/wav2vec2-large-xlsr-german"
model_name = "/share/w2v/wav2vec2-large-xlsr-german-sm"
print(validate_model(model_name, based_on))

exit(0)


supported_models = ['facebook/wav2vec2-large-xlsr-53-german',
                        'maxidl/wav2vec2-large-xlsr-german',
                        'marcel/wav2vec2-large-xlsr-53-german',
                        'flozi00/wav2vec-xlsr-german',
                        'marcel/wav2vec2-large-xlsr-german-demo',
                        'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German']
parser_supported_models_str = ["\t" + model for model in supported_models]
parser_supported_models_str = "Supported Models: \r\n" + "\r\n".join(parser_supported_models_str)


#parser = argparse.ArgumentParser(description="Evaluate Wav2Vec", formatter_class=RawTextHelpFormatter)
#parser.add_argument('--model_name', '-m', required=True,
#                    help=parser_supported_models_str, type=in_list(supported_models, "\r\nInvalid Model Name: "))

#args = parser.parse_args()
#model_name = args.model_name
#model_name = "flozi00/wav2vec-xlsr-german"
for model_name in supported_models:
    try:
        print(validate_model(model_name))
    except Exception as e:
        error = "\t".join([model_name, "error", str(e)])
        print(error)
