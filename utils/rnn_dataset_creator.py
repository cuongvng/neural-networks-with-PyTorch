import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# Create my own dataset class that reads data from csv, partitions it.
class ForexDataset(Dataset):
	def __init__(self, csv_file, num_steps=20):
		super(ForexDataset, self).__init__()
		self.data = self.get_sequential_data(df=pd.read_csv(csv_file), num_steps=num_steps)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	def get_sequential_data(self, df:pd.DataFrame, num_steps:int):
		ask_price = pd.Series(df["Ask"])

		data = []
		for i in range(0, len(ask_price)-num_steps-1):
			X = ask_price.values[i:i+num_steps]
			y = ask_price.values[i+num_steps]
			data.append((torch.tensor(X),torch.tensor(y)))
		return data

if __name__ == "__main__":
	dataset = ForexDataset(csv_file="data/EURUSD.csv")
	dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

	for (x, y) in iter(dataloader):
		print( x, y)