import torch
import numpy as np

from operator import itemgetter
from torch.utils.data import TensorDataset, Dataset

class EmbeddingsDataset(Dataset):
    def __init__(self,seqs,labels):

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")
        self.labels = labels
        self.seqs = seqs
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        return self.seqs[index], self.labels[index]
    

class TimeSeriesDataset(Dataset):
    def __init__(self,seqs,labels):

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")
        self.labels = labels
        self.seqs = np.asarray(seqs,dtype=np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        return self.seqs[index], self.labels[index]

def collate_embeddings(batch):
	"""
    :returns 
        seqs (FloatTensor) - 3D BACTCH SIZE X max_lenght X num_features(100=dim embeddings)
        lenghts(LongTensor) - 1D of batch size
        labels (LongTensor) - 1D of batch size
    """

	max_len = max([seq[0].shape[0] for seq in batch])
	new_seqs=[]
	for seq in batch:
		label = seq[1]
		length_ = seq[0].shape[0] 
		pad = np.zeros((max_len-seq[0].shape[0],seq[0].shape[1]))
		new_seq = np.concatenate((seq[0], pad), axis=0)

		new_seqs.append((new_seq,length_,label))
		
	new_sorted_seqs = sorted(new_seqs, key=itemgetter(1),reverse=True)

	seqs_tensor = torch.FloatTensor(np.array([seq[0] for seq in new_sorted_seqs]))
	lengths_tensor = torch.LongTensor(np.array([seq[1] for seq in new_sorted_seqs]))
	labels_tensor = torch.LongTensor(np.array([seq[2] for seq in new_sorted_seqs]))

	return seqs_tensor,labels_tensor
