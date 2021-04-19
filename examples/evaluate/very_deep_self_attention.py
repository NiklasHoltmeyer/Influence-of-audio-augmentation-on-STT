from audioengine.model.very_deep_self_attention.helper import create_model_test, get_data_from_df
from audioengine.corpus.dataset import CommonVoice
from audioengine.model.very_deep_self_attention.helper import create_model, create_tf_dataset
from audioengine.model.very_deep_self_attention.embedding import VectorizeChar
import tensorflow as tf
from audioengine.metrics.wer import Jiwer
import os
from tqdm import tqdm
from audioengine.metrics.wer import Jiwer

## Load-Model

core_count = os.cpu_count()
common_voice_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
audio_format = "mp3"
common_voice = CommonVoice(common_voice_path)
test_df = common_voice.load_dataframe(type="test")
input_key, output_key = "path", "sentence"

test_data = get_data_from_df(test_df, input_key, output_key)

batch_size = 32
max_target_len = 172
checkpoint_dir = "/share/train/deepselfatt/16042021_15_33_31/cp/"

model = create_model(max_target_len, len(test_data))
vectorizer = VectorizeChar(max_target_len)
idx_to_char = vectorizer.get_vocabulary()

val_ds = create_tf_dataset(test_data, audio_format, vectorizer, batch_size)

latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest_cp)


## Test Model

def idx_to_ch(idxs, idx_to_char):
    y = "".join([str(idx_to_char[idx]) for idx in idxs]).replace("-", "").replace("<", "").replace(">",
                                                                                                   "").strip().lower()
    return y


def batch_idx_to_str(batch, key, idx_to_char):
    batch_idxs = batch[key].numpy()
    sentences = [idx_to_ch(idxs, idx_to_char) for idxs in batch_idxs]
    return sentences


def batch_predict(model, batch):
    targets = batch_idx_to_str(batch, "target", idx_to_char)
    sources = batch["source"]
    preds_idx = model.generate(sources, target_start_token_idx=2).numpy()
    predictions = [idx_to_ch(pred, idx_to_char) for pred in preds_idx]
    return targets, predictions


test = None

wer = Jiwer()
sentence_stacked = []
transcriptions_stacked = []

for idx, batch in enumerate(tqdm(val_ds)):
    targets, predictions = batch_predict(model, batch)
    transcriptions_stacked.extend(predictions)
    sentence_stacked.extend(targets)

    if idx % 71 == 0:  # if idx % 13 == 0:
        wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
        sentence_stacked, transcriptions_stacked = [], []

result_tsv = wer.to_tsv(prefix="Very_Deep_Self_attention 72Ep")
print(result_tsv)
