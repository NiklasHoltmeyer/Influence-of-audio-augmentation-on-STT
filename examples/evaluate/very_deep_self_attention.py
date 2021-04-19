import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.metrics.wer import Jiwer
from audioengine.model.pretrained.silero import Silero


#def validate_model(model_language):
#    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
#    silero = Silero(model_language, device=device)
#    transform = transforms.Compose(silero.transformations())
#    dataset = Dataset("torch").CommonVoice("/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
                                           #shuffle=False, transform=transform, type="test")

#    core_count = os.cpu_count()

#    dataloader = DataLoader(dataset, batch_size=16, num_workers=os.cpu_count(),
                            #collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

#    wer = Jiwer()
#    sentence_stacked = []
#    transcriptions_stacked = []
#    for idx, (speeches, sentences) in enumerate(tqdm(dataloader)):
#        transcriptions = silero.predict(speeches)
#        transcriptions_stacked.extend(transcriptions)
#        sentence_stacked.extend(sentences)

#        if idx % 13 == 0:
#            wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
#            sentence_stacked, transcriptions_stacked = [], []

#    return wer.to_tsv(prefix="Silero-" + model_language)
#from audioengine.model.very_deep_self_attention.helper import create_model_test
#batch_size=32
#cp_path="/share/train/deepselfatt/16042021_15_33_31/cp/"
#create_model_test(test_data, max_target_len, audio_format, batch_size, cp_path)




