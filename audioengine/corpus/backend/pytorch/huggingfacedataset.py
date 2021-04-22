from datasets import Dataset
class HuggingfaceDataset:
    def from_dataframe(self, data_frame, input_key, target_key, transform=None, features=None, **kwargs):
        return Dataset.from_pandas(data_frame)
