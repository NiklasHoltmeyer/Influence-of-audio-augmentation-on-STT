import argparse
from argparse import RawTextHelpFormatter, ArgumentTypeError
from torchvision import transforms
from audioengine.model.pretrained.silero import Silero
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


def validate_model(model_language):
    silero = Silero(model_language)
    transform = transforms.Compose(silero.transformations())
    dataset = Dataset("torch").CommonVoice("/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
                                           shuffle=False, transform=transform, type="test")

    core_count = os.cpu_count()

    dataloader = DataLoader(dataset, batch_size=16, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    wer = Jiwer()
    sentence_stacked = []
    transcriptions_stacked = []
    for idx, (speeches, sentences) in enumerate(tqdm(dataloader)):
        transcriptions = silero.predict(speeches)
        transcriptions_stacked.extend(transcriptions)
        sentence_stacked.extend(sentences)

        if idx % 13 == 0:
            wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
            sentence_stacked, transcriptions_stacked = [], []

    return wer.to_tsv(prefix="Silero-" + model_language)


def in_list(_list, exception_text):
    def __call__(item):
        if not item in _list:
            raise ArgumentTypeError(exception_text + item)
        return item

    return __call__


supported_models = ['de', 'es', 'en']
parser_supported_models_str = ["\t" + model for model in supported_models]
parser_supported_models_str = "Supported Models: \r\n" + "\r\n".join(parser_supported_models_str)

parser = argparse.ArgumentParser(description="Evaluate Silero", formatter_class=RawTextHelpFormatter)
parser.add_argument('--model_language', '-l', required=True,
                    help=parser_supported_models_str, type=in_list(supported_models, "\r\nInvalid Model Language: "))

args = parser.parse_args()
model_language = args.model_language
