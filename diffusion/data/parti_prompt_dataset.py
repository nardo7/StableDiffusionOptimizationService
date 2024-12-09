from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PartiPromptDataset(Dataset):
    """
    The dataset class for the PartiPrompt dataset. It will load the dataset into a dataframe 
    and access it to get the samples.
    """

    def __init__(self, path: str):
        """
        Initialize the dataset with the given path

        Args:
            path (str): The path to the dataset
        """
        self.path = path
        self.data_table = pd.read_csv(path, sep="\t", engine='c', memory_map=True, dtype={"Prompt": np.str_})  

    def __len__(self):
        """
        Get the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.data_table)

    def __getitem__(self, idx):
        """
        Get the item at the given index

        Args:
            idx (int): The index of the item

        Returns:
            Any: The item at the given index
        """
        return {"Prompt": self.data_table.at[idx, "Prompt"]}