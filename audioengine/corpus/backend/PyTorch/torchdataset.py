from torch.utils.data import DataLoader

from audioengine.corpus.backend.PyTorch import DataframeDataset


class TorchDataset:
    def from_dataframe(self, data_frame, input_key, target_key, transform=None, **kwargs):
        """

        Args:
            data_frame: Data
            input_key: Dataframe Column Name for Input
            target_key: Dataframe Column Name for Targets
            transform: (Composed) Transform(s)
            **kwargs: batch_size, shuffle, num_workers

        Returns:
            DataLoader
        """
        batch_size = kwargs.get("batch_size", 32)
        shuffle = kwargs.get("shuffle", False)
        num_workers = kwargs.get("num_workers", False)

        ds = DataframeDataset(data_frame, input_key, target_key, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
