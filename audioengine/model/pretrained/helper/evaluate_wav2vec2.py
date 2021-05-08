import argparse
from argparse import RawTextHelpFormatter, ArgumentTypeError
from pathlib import Path

import torch
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.corpus.util.text import save_settings
from audioengine.logging.logging import defaultLogger
from audioengine.metrics.wer import Jiwer
from torchvision import transforms
from audioengine.model.pretrained.wav2vec2 import wav2vec2
from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from torch.utils.data import DataLoader
import os
import time
import pandas as pd
from tqdm import tqdm
# from tqdm.auto import tqdm

from audioengine.transformations.backend.pytorch.texttransformations import ToUpper

#    return wer.to_tsv(prefix=model_name, suffix=str(time.time()-start_time)).replace(".", ",")

logger = defaultLogger()


def evaluate(model_name, settings):
    assert "dataset" in settings.keys(), "DataSet Settings needed!"

    w2v = wav2vec2(model_name)
    settings["dataset"]["transform"] = w2v.transformation()

    logger.debug("*" * 72)
    logger.debug(model_name, "loaded.")

    (_, _), (ds, ds_info) = Dataset("torch").from_settings(settings["dataset"])

    core_count = os.cpu_count()
    dataloader = DataLoader(ds, batch_size=20, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    return _run_eval(w2v, dataloader, settings)


def _run_eval(w2v, dataloader, settings):
    wer = Jiwer()

    sentence_stacked = transcriptions_stacked = []
    sentences_full = transcriptions_full = []

    start_time = time.time()

    eval_settings = settings.get("eval", {})
    threads = eval_settings.get("num_workers", os.cpu_count())

    infos = {}

    for idx, (speeches, sentences) in enumerate(tqdm(dataloader)):
        transcriptions = w2v.predict(speeches)
        transcriptions_stacked.extend(transcriptions)
        sentence_stacked.extend(sentences)

        if idx % 97 == 0:  # 97 71
            wer.add_batch(sentence_stacked, transcriptions_stacked, threads)
            transcriptions_full.extend(transcriptions_stacked)
            sentences_full.extend(sentence_stacked)
            sentence_stacked, transcriptions_stacked = [], []

    infos["elapsed_time"] = end_time = time.time() - start_time
    infos["wer"] = {"score":wer.calc()}

    path = eval_settings.get("path", None)
    skip_wordwise_wer = eval_settings.get("skip_wordwise_wer", False)

    result = pd.DataFrame({"sentences": sentences_full, "transcriptions": transcriptions_full})

    if not skip_wordwise_wer:
        result["wer"] = _per_prediction_wer(sentences_full, transcriptions_full)
        assert len(sentences_full) == len(transcriptions_full) == len(result["wer"])

    if path:
        decimal_symbol = eval_settings.get("decimal", ".")
        sep_symbol = eval_settings.get("sep", "\t")

        csv_path = Path(f"{path}/{w2v.model_name}/result.tsv")
        json_path = Path(f"{path}/{w2v.model_name}/result_infos.json")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = str(csv_path.resolve())

        result.to_csv(csv_path, encoding="UTF8", sep=sep_symbol, index=False, decimal=decimal_symbol)
        infos["wer"]["median"] = result.median()
        infos["wer"]["mean"] = result.mean()
        infos["wer"]["min"] = result.min()
        infos["wer"]["max"] = result.max()
        infos["wer"]["var"] = result.var()
        infos["wer"]["std"] = result.var()

        infos["dataset"] = settings.get("dataset", {})
        save_settings(json_path, infos)

    return infos, result

def _per_prediction_wer(sentences, predictions):
    def _calc_wer(sentence, transcript):
        _jiwer = Jiwer()
        _jiwer.add(sentence, transcript)
        return _jiwer.calc()

    _wers = [_calc_wer(sentence, transcript) for sentence, transcript in tqdm(zip(sentences, predictions))]
    return _wers

