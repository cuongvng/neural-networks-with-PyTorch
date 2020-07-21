# neural-networks-with-PyTorch
PyTorch implementation of popular CNNs and RNNs.


## Things I have learned:
- When creating custom dataset using `torch.utils.data.Dataset`, the data should be in complete form, i.e, `list`, `ndarray`, etc., **NOT** `generator`.
The reason is `generator` only allows to iterate over the dataset **one time**, thus the `StopIteration` is likely to be raised by `torch.utils.data.DataLoder` when loading the dataset.
- For huge dataset, it is more memory efficient to keep the whole file paths (just string) as the complete form of data, then use loading the real data by batch in the `__getitem__` method. 
