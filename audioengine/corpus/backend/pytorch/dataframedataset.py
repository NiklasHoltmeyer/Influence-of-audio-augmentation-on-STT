import torch
from torch.utils.data import Dataset


class DataframeDataset(Dataset):
    """Load Pytorch Dataset from Dataframe
    
    """

    def __init__(self, data_frame, input_key, target_key, transform=None, features=None):
        self.data_frame = data_frame
        self.input_key = input_key
        self.target_key = target_key
        self.inputs = self.data_frame[input_key]
        self.targets = self.data_frame[target_key]
        self.transform = transform
        self.features = [input_key, target_key] if features is None else features
        self.len = len(self.inputs)

    def __len__(self):
        return self.len

    def __str__(self):
        return str(self.info())

    def info(self):
        info = {
            'features': self.features,
            'num_rows': len(self)
        }
        return info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = {
            self.input_key: self.inputs[idx],
            self.target_key: self.targets[idx]
        }

        if self.transform:
            return self.transform(data)

        return data

    @staticmethod
    def collate_fn(input_key, output_key):
        def __call__(batch):
            speeches = [data[input_key] for data in batch]
            sentences = [data[output_key] for data in batch]
            return speeches, sentences

        return __call__


#    @staticmethod
#    def collate_fn(batch):
#        return [(data["input"], data["output"]) for data in batch]


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    data = [("x1", "y2", "A3"), ("x1", "y2", "b3"), ("x1", "y2", "c3"), ("x1", "y2", "d3")]
    df = pd.DataFrame(data, columns=['input', 'target', 'random'])
    print(df.head())

    ds = DataframeDataset(data_frame=df, input_key="input", target_key="target", transform=None)
    print("*" * 40)
    print("Len:", len(ds))
    print("Ds", ds)

    # loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=4)  # collate_fn=DataframeDataset.collate_fn
    # print(loader)
    print("o" * 33)
    for idx in range(0, len(ds)):
        data = ds[idx]
        print(idx, "->", data)
        # pass
#    for idx, (x, y) in enumerate(loader):
#        print("x", x, "\t", "y", y)

# asd
