import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import itertools

# Create my own dataset class that reads data from csv, partitions it.
class ForexDataset(Dataset):
	def __init__(self, csv_file, num_steps=20):
		super(ForexDataset, self).__init__()
		self.datagen = self.get_sequential_data(df=pd.read_csv(csv_file), num_steps=num_steps)

	def __len__(self):
		return len(list(self.datagen))

	def __getitem__(self, idx):
		return next(itertools.islice(self.datagen, idx))

	def get_sequential_data(self, df:pd.DataFrame, num_steps:int):
		ask_price = pd.Series(df["Ask"])

		for i in range(0, len(ask_price)-num_steps-1):
			X = ask_price.values[i:i+num_steps]
			y = ask_price.values[i+num_steps]
			yield torch.tensor(X), torch.tensor(y)

if __name__ == "__main__":
	dataset = ForexDataset(csv_file="data/EURUSD.csv")
	dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

	for (x, y) in iter(dataloader):
		print( x, y)
	# Error: StopIteration