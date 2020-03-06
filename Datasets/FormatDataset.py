import unicodedata

from pandas import DataFrame
from torch.utils.data import Dataset


class FormatDataset(Dataset):
    def __init__(self, df: DataFrame, name_hdr: str, format_hdr: str):
        self.name_hdr = name_hdr
        self.format_hdr = format_hdr
        self.data_frame = df[[name_hdr, format_hdr]]
        self.data_frame[name_hdr] = df[name_hdr].apply(
            lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame[self.name_hdr].iloc[index], self.data_frame[self.format_hdr].iloc[index]
