import os
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.corpus.util.text import save_settings
from audioengine.logging.logging import defaultLogger
from audioengine.metrics.wer import Jiwer
from audioengine.model.pretrained.wav2vec2 import wav2vec2

logger = defaultLogger()


def evaluate(settings):
    assert "dataset" in settings.keys(), "DataSet Settings needed!"
    assert "eval" in settings.keys(), "DataSet Settings needed!"
    model_name = settings["eval"]["model_name"]
    model_based_on_name = settings["eval"].get("model_based_on", None)
    w2v = wav2vec2(model_name, based_on=model_based_on_name)
    settings["dataset"]["transform"] = w2v.transformation()

    logger.debug("*" * 72)
    logger.debug(model_name + " loaded.")

    (_, _), (ds, ds_info) = Dataset("torch").from_settings(settings["dataset"])

    dataloader = DataLoader(ds, batch_size=20, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    return _run_eval(w2v, dataloader, settings)


def _run_eval(w2v, dataloader, settings):
    wer = Jiwer()

    sentence_stacked = transcriptions_stacked = []
    sentences_full = transcriptions_full = []

    start_time = time.time()

    eval_settings = settings.get("eval", {})
    threads = eval_settings.get("num_workers", os.cpu_count() * 2)

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

    infos["elapsed_time"] = time.time() - start_time

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

        infos["wer"] = {"score":    wer.calc(),
                        "median":   result["wer"].median(),
                        "mean":     result["wer"].mean(),
                        "min":      result["wer"].min(),
                        "max":      result["wer"].max(),
                        "var":      result["wer"].var()}

        infos["dataset"] = settings.get("dataset", {})
        if "transform" in infos["dataset"].keys():
            del settings["dataset"]["transform"]
        save_settings(json_path, infos)
    else:
        infos["wer"] = {"score": wer.calc()}

    return infos, result


def _per_prediction_wer(sentences, predictions):
    def _calc_wer(sentence, transcript):
        _jiwer = Jiwer()
        _jiwer.add(sentence, transcript)
        return _jiwer.calc()

    _wers = [_calc_wer(sentence, transcript) for sentence, transcript in tqdm(zip(sentences, predictions))]
    return _wers


if __name__ == "__main__":
    cv_test_full = {
        "base_path": "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de",
        "shuffle": False,
        "validation_split": None,
        "type": "test",
        "min_target_length": 2,
    }

    eval_settings = {  # eval
        "path": "/share/notebook/eval_results",
        "decimal": ",",
        "model_name": "maxidl/wav2vec2-large-xlsr-german"
    }

    settings = {
        "dataset": {"val_settings": [cv_test_full], "train_settings": None},
        "eval": eval_settings
    }

    infos, result = evaluate(settings)
