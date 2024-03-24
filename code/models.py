import torch
from torch import nn
from torch.nn import functional as F


class ConvolutionNER(nn.Module):
  def __init__(self, dim_input):
    super(ConvolutionNER, self).__init__()

    self.conv1 = nn.Conv1d(in_channels=dim_input,out_channels=32,kernel_size=3,padding=2,stride=1)
    self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=2,stride=1)
    self.conv3 = nn.Conv1d(in_channels=64,out_channels=96,kernel_size=3,padding=2,stride=1)
    self.globalpooling= nn.AdaptiveMaxPool1d(1)
    self.flatten = nn.Flatten()
    self.dropout1= nn.Dropout(0.2)
    self.fc1 = nn.Linear(in_features=96,out_features=512)
    self.fc = nn.Linear(in_features=512,out_features=2)
    
  def forward(self, input):

    seqs = input
    #cnn takes input of shape (batch_size, channels, seq_len)
    out = seqs.permute(0,2,1)
    out = F.relu(self.conv1(out))
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv3(out))
   
    out = self.globalpooling(out)
    out = self.flatten(out)
    out = F.relu(self.fc1(out))
    out = self.dropout1(out)
    out = self.fc(out)
    
    return out