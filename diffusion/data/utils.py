from torch.utils.data import Dataset, random_split

def create_sub_dataset(dataset: Dataset, sub_size: int) -> Dataset:
    """
    Create a sub-dataset with a specified number of examples from another dataset.

    Args:
        dataset (datasets.Dataset): The original dataset.
        sub_size (int): The number of examples to include in the sub-dataset.

    Returns:
        datasets.Dataset: A new dataset containing the specified number of examples.
    """
    # Use train_test_split to create the sub-dataset
    sub_dataset, _ = random_split(dataset, [sub_size, len(dataset) - sub_size])
    return sub_dataset

