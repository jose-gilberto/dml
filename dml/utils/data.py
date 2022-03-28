from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset


class DiabetesDataset(Dataset):

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        transform = None,
        target_transform = None
    ) -> None:
        super().__init__()

        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        instance = self.data.iloc[index, 1:].values
        label = self.data.iloc[index, 0]

        if self.transform:
            instance = self.transform(instance)

        if self.target_transform:
            label = self.target_transform(label)

        return torch.tensor(instance).float(), torch.tensor([label]).float()