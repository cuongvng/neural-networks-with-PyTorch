import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# Create my own dataset class that reads data from csv, partitions it.
class ForexDataset(Dataset):
	def __init__(self, csv_file, num_steps=20, train=True, train_size=0.8):
		super(ForexDataset, self).__init__()
		df = pd.read_csv(csv_file)

		train_size = int(train_size * df.shape[0])
		if train:
			df = df.iloc[0:train_size]
		else:
			df = df.iloc[train_size:]
		self.data = self.get_sequential_data(df=df, num_steps=num_steps)

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
	data_train = ForexDataset(csv_file="../RNNs/data/EURUSD.csv", train=True)
	data_test = ForexDataset(csv_file="../RNNs/data/EURUSD.csv", train=False)
	print(len(data_train), len(data_test))

	dataloader = DataLoader(dataset=data_train, batch_size=64, shuffle=False)

	for (x, y) in iter(dataloader):
		print( x, y)