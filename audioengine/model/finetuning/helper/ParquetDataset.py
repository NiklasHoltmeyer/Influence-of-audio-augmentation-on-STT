from multiprocessing import Pool
from tqdm.auto import tqdm
import torch
from pathlib import Path
import pandas as pd


class ParquetDataset(torch.utils.data.Dataset):

    def __init__(self, data_args, split='train'):
        super().__init__()
        assert split in {'train', 'eval'}
        self.split = split
        resampled_data_dir = Path(data_args.dataset_path)
        self.path = Path(f'{resampled_data_dir}/{data_args.dataset_config_name}.{split}.parquet')

        assert self.path.exists()

        df = pd.read_parquet(self.path)
        self.labels = [x.tolist() for x in df['labels'].tolist()]
        self.paths = df['path'].tolist()
        self.max_input_length_quantile = .98
        self.max_input_length = None
        self.dataloader_num_workers = data_args.preprocessing_num_workers

        if split == 'train':
            with Pool(self.dataloader_num_workers) as p:
                self.input_seq_lengths = list(
                    tqdm(p.imap(get_input_len, self.paths), total=len(self.paths), miniters=100,
                         desc='getting train input lengths'))
            self.max_input_length = torch.tensor(self.input_seq_lengths).float().quantile(
                self.max_input_length_quantile).int().item()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inputs = load_speech(self.paths[idx])
        if self.split == 'train':
            inputs = inputs[:self.max_input_length]
        label = self.labels[idx]
        return {'input_values': inputs, 'labels': label}


def load_speech(f):
    return torch.load(f).squeeze().tolist()


def get_input_len(f):
    t = torch.load(f).squeeze().tolist()
    return len(t)
