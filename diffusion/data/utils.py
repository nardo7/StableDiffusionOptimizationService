import datasets

def create_sub_dataset(dataset: datasets.Dataset, sub_size: int) -> datasets.Dataset:
    """
    Create a sub-dataset with a specified number of examples from another dataset.

    Args:
        dataset (datasets.Dataset): The original dataset.
        sub_size (int): The number of examples to include in the sub-dataset.

    Returns:
        datasets.Dataset: A new dataset containing the specified number of examples.
    """
    # Calculate the proportion of the dataset to include in the sub-dataset
    sub_proportion = sub_size / len(dataset)

    # Use train_test_split to create the sub-dataset
    sub_dataset = dataset.train_test_split(test_size=1 - sub_proportion, seed=42)['train']

    return sub_dataset