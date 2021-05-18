import os
import time
from argparse import ArgumentTypeError
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from audioengine.corpus.backend.pytorch.dataframedataset import DataframeDataset
from audioengine.corpus.dataset import Dataset  # dataset.Dataset
from audioengine.metrics.wer import Jiwer
from audioengine.model.pretrained.wav2vec2 import wav2vec2


def validate_model(model_name, based_on=None):
    w2c = wav2vec2(model_name, based_on=based_on)
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

    core_count = os.cpu_count()  # batch-size -> 20
    dataloader = DataLoader(ds, batch_size=16, num_workers=os.cpu_count(),
                            collate_fn=DataframeDataset.collate_fn("speech", "sentence"))

    wer = Jiwer()
    sentence_stacked = []
    transcriptions_stacked = []
    start_time = time.time()
    for idx, (speeches, sentences) in enumerate(tqdm(dataloader, desc=model_name)):
        transcriptions = w2c.predict(speeches)
        transcriptions_stacked.extend(transcriptions)
        sentence_stacked.extend(sentences)

        if idx % 97 == 0:  # 97 71
            wer.add_batch(sentence_stacked, transcriptions_stacked, core_count)
            sentence_stacked, transcriptions_stacked = [], []

    return wer.to_tsv(prefix=model_name, suffix=str(time.time() - start_time)).replace(".", ",")


def in_list(_list, exception_text):
    def __call__(item):
        if not item in _list:
            raise ArgumentTypeError(exception_text + item)
        return item

    return __call__


nh_proc = "unknownTransformer/wav2vec2-large-xlsr-german"

jobs = [
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_2/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-5000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-8000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-7000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_sm_nh/checkpoint-6000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1_conf/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_s_1_conf/checkpoint-6000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-4000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_small_noaug_1/checkpoint-2000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-3000', nh_proc),
#    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-4000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-10000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-5000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-11000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-8000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-2000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-7000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-6000', nh_proc),
    ('/share/modelle/no_aug/models/cv_md_noaug/checkpoint-9000', nh_proc),
    ('/share/modelle/no_aug/models/cv_s_1/checkpoint-6000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-3000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-4000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-10000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-5000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-8000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-2000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-7000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-6000', nh_proc),
    ('/share/modelle/noise/models/cv_sm_real_noise/checkpoint-9000', nh_proc),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-18000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-18000'),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-19000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-19000'),
    ('/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-17000',
     '/share/modelle/base/models/run_g_f_p_1_resume/checkpoint-17000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-3000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-3000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-1000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-1000'),
    (
    '/share/modelle/base/models/run_g_f_i_1/checkpoint-2000', '/share/modelle/base/models/run_g_f_i_1/checkpoint-2000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-5000', '/share/modelle/base/models/run_k_3/checkpoint-5000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-1000', '/share/modelle/base/models/run_k_3/checkpoint-1000'),
    ('/share/modelle/base/models/run_k_3/checkpoint-2000', '/share/modelle/base/models/run_k_3/checkpoint-2000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-22000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-22000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-23000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-23000'),
    ('/share/modelle/base/models/run_pro_idleback/checkpoint-24000',
     '/share/modelle/base/models/run_pro_idleback/checkpoint-24000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-3000', '/share/modelle/base/models/run_k_1/checkpoint-3000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-4000', '/share/modelle/base/models/run_k_1/checkpoint-4000'),
    ('/share/modelle/base/models/run_k_1/checkpoint-5000', '/share/modelle/base/models/run_k_1/checkpoint-5000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-8000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-8000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-1000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-1000'),
    ('/share/modelle/base/models/run_pro_750_wu/checkpoint-9000',
     '/share/modelle/base/models/run_pro_750_wu/checkpoint-9000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-8000',
     '/share/modelle/base/models/run_pro_500_wu/checkpoint-8000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-7000',
     '/share/modelle/base/models/run_pro_500_wu/checkpoint-7000'),
    ('/share/modelle/base/models/run_pro_500_wu/checkpoint-9000', '/share/modelle/base/models/run_pro_500_wu/checkpoint - 9000'),
]

results = []
failed = []

##/share/modelle/vf_cv_small/results.tsv
##/share/modelle/vf_cv_small/failed.tsv

def already_run(model_name, file_path):
    with open(file_path, "r+") as f:
        for line in f.readlines():
            line_model = line.split("\t")[0]
            if model_name in line_model:
                return True
    return False

result_path = "/share/modelle/check_results_checkpoints_bruteforce.tsv"
failed_path = "/share/modelle/check_failed.tsv"

for model_name, based_on in tqdm(
        jobs, desc="troll"):
    try:
        # base_on = str(Path(model_name).parent.resolve())
        # base_on = "/share/datasets/wav2vec2-large-xlsr-german-vf_nh"
        if not already_run(model_name, result_path):
            result_tsv = validate_model(model_name, based_on=based_on)

            with open(result_path, "a+") as f:
                f.write("\t".join([result_tsv]))
                f.write("\n")

            results.append(result_tsv)
    except Exception as e:
        print(e)
        with open(failed_path, "a+") as f:
            f.write("\t".join([model_name, "2" + str(e)]))
            f.write("\n")
        failed.append(model_name)

print("*" * 72)
print("*" * 72)
print("*" * 72)
for result in results:
    print(result)
print("-" * 72)
print(results)
print("-" * 72)
for fail in failed:
    print(fail)
print("-" * 72)
print(failed)
print("*" * 72)
print("*" * 72)
print("*" * 72)

supported_models = ['facebook/wav2vec2-large-xlsr-53-german',
                    'maxidl/wav2vec2-large-xlsr-german',
                    'marcel/wav2vec2-large-xlsr-53-german',
                    'flozi00/wav2vec-xlsr-german',
                    'marcel/wav2vec2-large-xlsr-german-demo',
                    'MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-German']
parser_supported_models_str = ["\t" + model for model in supported_models]
parser_supported_models_str = "Supported Models: \r\n" + "\r\n".join(parser_supported_models_str)

# parser = argparse.ArgumentParser(description="Evaluate Wav2Vec", formatter_class=RawTextHelpFormatter)
# parser.add_argument('--model_name', '-m', required=True,
#                    help=parser_supported_models_str, type=in_list(supported_models, "\r\nInvalid Model Name: "))

# args = parser.parse_args()
# model_name = args.model_name
# model_name = "flozi00/wav2vec-xlsr-german"
# for model_name in supported_models:
#    try:
#        print(validate_model(model_name))
#    except Exception as e:
#        error = "\t".join([model_name, "error", str(e)])
#        print(error)
