import os
import pandas as pd


class Text:
    @staticmethod
    def read_csv(*path, **kwargs):
        path = os.path.join(*path)

        if not os.path.exists(path):
            raise Exception(f"File not Found\n {path}")

        return pd.read_csv(path, **kwargs)

