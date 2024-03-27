import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


  
class ConvolutionalNERGRU(nn.Module):
	def __init__(self):
		super(ConvolutionalNERGRU, self).__init__()

		self.conv1 = nn.Conv1d(in_channels=100,out_channels=32,kernel_size=3,padding=2,stride=1)
		self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=2,stride=1)
		self.conv3 = nn.Conv1d(in_channels=64,out_channels=96,kernel_size=3,padding=2,stride=1)
		self.globalpooling= nn.AdaptiveMaxPool1d(1)
		self.flatten = nn.Flatten()
		self.dropout1= nn.Dropout(0.2)

		self.gru = nn.GRU(104, 256, dropout=0.2, batch_first=True)
		self.sigmoid = nn.ReLU()
		self.hiddenLayer = nn.Linear(352, 2)

	def forward(self, input):
		embed,seqs = input
		out_gru, _ = self.gru(seqs)


		#cnn takes input of shape (batch_size, channels, seq_len)
		out_embed = embed.permute(0,2,1)
		out_embed = F.relu(self.conv1(out_embed))
		out_embed = F.relu(self.conv2(out_embed))
		out_embed = F.relu(self.conv3(out_embed))
		out_embed = self.globalpooling(out_embed)
		out_embed = self.flatten(out_embed)

		output = torch.concat((out_embed,out_gru[:,-1,:]),dim=1)

		output = self.hiddenLayer(output)
		output = self.sigmoid(output)

		return output

class GRU(nn.Module):
	def __init__(self):
		super(GRU, self).__init__()

		self.gru = nn.GRU(104, 256, dropout=0.2, batch_first=True)
		self.sigmoid = nn.Sigmoid()
		self.hiddenLayer = nn.Linear(256, 2)

	def forward(self, input):
		output, _ = self.gru(input)
		output = self.hiddenLayer(output[:, -1, :])
		output = self.sigmoid(output)

		return output

