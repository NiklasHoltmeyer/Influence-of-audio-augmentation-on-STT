import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_io as tfio

from audioengine.model.very_deep_self_attention.embedding import VectorizeChar
from audioengine.model.very_deep_self_attention.transformer import DisplayOutputs, Transformer, CustomSchedule


def load_libri(path):
    wavs = glob("{}/wavs/*.wav".format(path), recursive=True)

    id_to_text = {}
    with open(os.path.join(path, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[2]
            id_to_text[id] = text
    return wavs, id_to_text


def get_data_libri(wavs, id_to_text, maxlen):
    data = []
    for w in wavs:
        id = w.split("/")[-1].split(".")[0]
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data


def path_to_audio_wav(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def path_to_audio_mp3(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tfio.audio.decode_mp3(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def get_data_from_df(df, input_key="path", output_key="sentence"):
    paths = df.pop("path").values.astype("str")
    texts = df.pop("sentence").values.astype("str")
    data = [{'audio': item[0], 'text': item[1]} for item in zip(paths, texts)]
    return data


def get_longest_string(df, key):
    return df[key].map(len).max()


def create_text_ds(data, vectorizer):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def create_audio_ds(data, audio_format="mp3"):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    if audio_format == "mp3":
        return audio_ds.map(path_to_audio_mp3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if audio_format == "wav":
        return audio_ds.map(path_to_audio_wav, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    raise Exception("Unknown Format")


def create_tf_dataset(data, audio_format, vectorizer, batch_size=4):
    audio_ds = create_audio_ds(data, audio_format)
    text_ds = create_text_ds(data, vectorizer)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def train_model(train_data, test_data, max_target_len, audio_format, batch_size, epochs, callbacks=[]):
    vectorizer = VectorizeChar(max_target_len)
    idx_to_char = vectorizer.get_vocabulary()

    ds = create_tf_dataset(train_data, audio_format, vectorizer, batch_size)
    val_ds = create_tf_dataset(test_data, audio_format, vectorizer, batch_size)

    batch = next(iter(val_ds))
    display_cb = DisplayOutputs(batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3)
    callbacks.append(display_cb)

    model = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=34,
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1,
    )

    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(ds),
    )

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model.fit(ds, validation_data=val_ds, callbacks=callbacks, epochs=epochs)
