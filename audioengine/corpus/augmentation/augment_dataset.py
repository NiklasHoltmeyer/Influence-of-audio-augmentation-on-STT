from audioengine.corpus.augmentation.helper import *

from audioengine.corpus.noise import Noise
from audioengine.corpus.voxforge import VoxForge

df = VoxForge("/share/datasets/vf_de").load_dataframe(shuffle="True", type="train")
noise_df = Noise("/share/datasets/FSD50K").load_dataframe()

augment_dataset(df, noise_df, output_dir="/share/datasets/vf_augment_train", output_subfolder="wav", sep="\t")
