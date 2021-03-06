from audioengine.corpus.dataset import CommonVoice
from audioengine.model.backend.tensorflow.training.callbacks import Callbacks
from audioengine.model.very_deep_self_attention.helper import get_data_from_df, get_longest_string, \
    create_model_training
from datetime import datetime

common_voice_path = "/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de"
result_path_path = "/share/train/"
audio_format = "mp3"
epochs = 100
batch_size = 32
logging_base_path = result_path_path + "deepselfatt/" + datetime.now().strftime("%d%m%Y_%H_%M_%S") + "/"
callbacks = Callbacks(batch_size).make(logging_base_path, "_".join(["dsa", str(epochs), str(batch_size)]))

common_voice = CommonVoice(common_voice_path)
train_df = common_voice.load_dataframe(type="train")  # train
test_df = common_voice.load_dataframe(type="test")
input_key, output_key = "path", "sentence"

max_target_len_train = get_longest_string(train_df, output_key)
max_target_len_test = get_longest_string(test_df, output_key)
max_target_len = max(max_target_len_train, max_target_len_test)

max_target_len = 157

train_data = get_data_from_df(train_df, input_key, output_key)
test_data = get_data_from_df(test_df, input_key, output_key)

model, callbacks, ds, val_ds = create_model_training(train_data, test_data, max_target_len, audio_format, batch_size,
                                                     epochs,
                                                     callbacks=callbacks)
history = model.fit(ds, validation_data=val_ds, callbacks=callbacks, epochs=epochs, workers=8)
