from pathlib import Path
import re
import torch
import torchaudio


def remove_special_characters(chars_to_ignore_regex):
    def __call__(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower() + " "
        return batch

    return __call__

def load_resample_save(resampled_data_dir, processor, target_sample_rate):
    def __call__(f):
        f = Path(f)
        new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
        if not new_path.exists():
            speech_array, sampling_rate = torchaudio.load(f)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sample_rate)
            speech_array_resampled = resampler(speech_array)
            input_values = processor(speech_array_resampled, sampling_rate=target_sample_rate).input_values
            input_values = torch.from_numpy(input_values).float().flatten()

            torch.save(input_values, new_path)
        return str(new_path)
    return __call__
