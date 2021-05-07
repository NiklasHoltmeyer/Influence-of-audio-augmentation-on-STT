from audioengine.corpus.dataset import Dataset
from audioengine.model.pretrained.wav2vec2 import wav2vec2

model_name = "facebook/wav2vec2-large-xlsr-53-german"
w2c = wav2vec2("facebook/wav2vec2-large-xlsr-53-german", skip_loading=True)

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
print(ds[0])
print("**")
print("type", type(ds))
print("Done")

#    def from_dataframe(self, data_frame, input_key, target_key, transform=None, features=None, **kwargs):
