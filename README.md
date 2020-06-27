# neural-networks-with-PyTorch
PyTorch implementation of popular CNNs and RNNs.


## Things I have learned:
- When creating custom dataset using `torch.utils.data.Dataset`, the data should be in complete form, i.e, `list`, `ndarray`, etc., not `generator`.
The reason is `generator` only allows to iterate over the dataset **one time**, thus the `StopIteration` is likely to be raised by `torch.utils.data.DataLoder` when loading the dataset.
