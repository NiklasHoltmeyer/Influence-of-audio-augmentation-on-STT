import torch
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.metrics.wer import Jiwer
from torchvision import transforms
from tqdm import tqdm
from audioengine.model.pretrained.wav2vec2 import wav2vec2
from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from torch.utils.data import DataLoader
import os


def validate_model(model_name):
    w2c = wav2vec2(model_name)
    transform = transforms.Compose(w2c.transformations())

    dataset = Dataset("torch").CommonVoice("/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de", shuffle=False,
                                           transform=transform, type="test")

    print("Dataset", dataset)
    print("Device", w2c.device)

    core_count = os.cpu_count()

    dataloader = DataLoader(dataset, batch_size=16, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    wer = Jiwer()
    s = []
    t = []
    for idx, (speeches, sentences) in enumerate(tqdm(dataloader)):
        transcriptions = w2c.predict(speeches)
        #t.extend(transcriptions)
        #s.extend(sentences)
        wer.add_batch(sentences, transcriptions)
        #if idx % 13 == 0:
#            wer.add_batch(s, t, core_count)
#            s, t = [], []
        #break

    print(wer.calc())
    print(wer.to_tsv_header(prefix=""))
    print(wer.to_tsv(prefix=model_name))
    torch.cuda.empty_cache()

model_name = "facebook/wav2vec2-large-xlsr-53-german"
validate_model(model_name)
#facebook/wav2vec2-large-xlsr-53-german	0.3188405797101449	94	37	7	0	16