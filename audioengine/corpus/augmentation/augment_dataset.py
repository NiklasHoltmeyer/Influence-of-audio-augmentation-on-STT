from audioengine.corpus.augmentation.helper import *

from audioengine.corpus.noise import Noise
from audioengine.corpus.commonvoice import CommonVoice

df = CommonVoice("/share/datasets/cv/de/cv-corpus-6.1-2020-12-11/de").load_dataframe(shuffle="True", type="train_small")
noise_df = Noise("/share/datasets/FSD50K").load_dataframe(shuffle="True")

augment_dataset(df, noise_df,
                output_dir="/share/datasets/cv_small_reverb",
                output_subfolder="wav",
                sep="\t",
                file_name="processed_train_small.tsv",
                filter_settings=filter_settings)
