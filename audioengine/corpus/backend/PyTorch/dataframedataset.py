import torch
from torch.utils.data import Dataset


class DataframeDataset(Dataset):
    """Load Pytorch Dataset from Dataframe
    
    """
    def __init__(self, data_frame, input_key, target_key, transform=None):
        self.data_frame = data_frame
        self.inputs = self.data_frame[input_key]
        self.targets = self.data_frame[target_key]
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.inputs[idx], self.targets[idx]

        if self.transform:
            return self.transform(data)

        return data
