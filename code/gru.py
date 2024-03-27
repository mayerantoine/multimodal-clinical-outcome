import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset,Dataset
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from plots import plot_learning_curves, plot_confusion_matrix
import numpy as np
from torch.nn import functional as F
from operator import itemgetter

MODE = "BOTH"  # Options: 'BOTH', 'TRAIN', 'TEST'
TARGET_VARIABLES = ["los_3", "los_7"]  # Options: 'mort_hosp', 'mort_icu', 'los_3', 'los_7'
NUMBER_OF_WORKERS = 0
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SIGMOID_THRESHOLD = 0.5

DATASET_FILE_PATH = "../output"
PATH_OUTPUT = "../output/model/"


class TimeSeriesDataset(Dataset):
	def __init__(self,seqs,embeddings,labels):

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		self.seqs = seqs #  torch view 24 X 104
		self.embed = embeddings
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self,index):
		return self.embed[index],self.seqs[index], self.labels[index]

def collate_embeddings(batch):
	"""
	'batch' is a list [(embed_1,labs_1, label_1), (embed_2,labs_2, label_2), ... , (embed_N,labs_N, label_N)]
	:returns 
		seqs (FloatTensor) - 3D BACTCH SIZE X max_lenght X num_features(100=dim embeddings)
		lenghts(LongTensor) - 1D of batch size
		labels (LongTensor) - 1D of batch size
	"""
	max_len = max([seq[0].shape[0] for seq in batch])
	new_seqs=[]
	for seq in batch:
		labs = seq[1]
		label = seq[2]
		length_ = seq[0].shape[0]
		pad = np.zeros((max_len-seq[0].shape[0],seq[0].shape[1]))
		new_seq = np.concatenate((seq[0], pad), axis=0)

		new_seqs.append((new_seq,length_,labs,label))
		
	#sorted list for the batch		
	new_sorted_seqs = sorted(new_seqs, key=itemgetter(1),reverse=True)
	

	embed_tensor = torch.FloatTensor(np.array([seq[0] for seq in new_sorted_seqs]))
	labs_tensor = torch.stack([seq[2] for seq in new_sorted_seqs])
	labels_tensor = torch.LongTensor(np.array([seq[3] for seq in new_sorted_seqs]))

	return (embed_tensor,labs_tensor),labels_tensor


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


def sigmoid_predict(output):
	results = []

	with torch.no_grad():
		for data in output:
			results.append(int(data[1] > SIGMOID_THRESHOLD))

	return torch.tensor(results)


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from utils.py
#
class Metrics:
	def __init__(self):
		self.value = 0
		self.average = 0
		self.sum = 0
		self.count = 0

	def update(self, value, n=1):
		self.value = value
		self.sum += self.value * n
		self.count += n
		self.average = self.sum / self.count


def calculate_accuracy(predicted, target):
	with torch.no_grad():
		batchSize = target.size(0)
		correct = predicted.eq(target).sum().item()

		return (correct / batchSize) * 100.0


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from utils.py and train_seizure.py
#
def train_model(model, device, dataLoader, criterion, optimizer, epoch):
	batchTime = Metrics()
	dataTime = Metrics()
	losses = Metrics()
	accuracy = Metrics()

	model.train()

	end = time.time()

	for i, (input, target) in enumerate(dataLoader):
		dataTime.update(time.time() - end)

		#input = input.to(device)
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)

		loss.backward()
		optimizer.step()

		batchTime.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(calculate_accuracy(sigmoid_predict(output), target), target.size(0))

		print(f'Epoch: [{epoch}][{i}/{len(dataLoader)}]\t'
		      f'Time {batchTime.value:.3f} ({batchTime.average:.3f})\t'
		      f'Data {dataTime.value:.3f} ({dataTime.average:.3f})\t'
		      f'Loss {losses.value:.4f} ({losses.average:.4f})\t'
		      f'Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})')

	return losses.average, accuracy.average


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from utils.py and train_seizure.py
#
def test_model(model, device, dataLoader, criterion):
	batchTime = Metrics()
	losses = Metrics()
	accuracy = Metrics()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()

		for i, (input, target) in enumerate(dataLoader):
			#input = input.to(device)

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			batchTime.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))

			predicted = sigmoid_predict(output)
			accuracy.update(calculate_accuracy(predicted, target), target.size(0))

			true = target.detach().cpu().numpy().tolist()
			predicted = predicted.detach().cpu().numpy().tolist()
			results.extend(list(zip(true, predicted)))

			print(f'Test: [{i}/{len(dataLoader)}]\t'
			      f'Time {batchTime.value:.3f} ({batchTime.average:.3f})\t'
			      f'Loss {losses.value:.4f} ({losses.average:.4f})\t'
			      f'Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})')

	return losses.average, accuracy.average, results


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from train_seizure.py
#
def model_runner(train, validation, test, targetVariable):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# model = GRU()
	model = ConvolutionalNERGRU()
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	#criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)

	if MODE.upper() == "BOTH" or MODE.upper() == "TRAIN":
		trainLoader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_embeddings,num_workers=NUMBER_OF_WORKERS)
		validationLoader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_embeddings,num_workers=NUMBER_OF_WORKERS)

		optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

		bestValidationAccuracy = 0.0
		trainLosses, trainAccuracies = [], []
		validationLosses, validationAccuracies = [], []

		for epoch in range(EPOCHS):
			trainLoss, trainAccuracy = train_model(model, device, trainLoader, criterion, optimizer, epoch)
			validationLoss, validationAccuracy, _ = test_model(model, device, validationLoader, criterion)

			trainLosses.append(trainLoss)
			validationLosses.append(validationLoss)

			trainAccuracies.append(trainAccuracy)
			validationAccuracies.append(validationAccuracy)

			if validationAccuracy > bestValidationAccuracy:
				bestValidationAccuracy = validationAccuracy
				torch.save(model, os.path.join(PATH_OUTPUT, f"{targetVariable}_GRU.pth"))

		plot_learning_curves(trainLosses, validationLosses, trainAccuracies, validationAccuracies,
							 f"{targetVariable.upper()} GRU")

	if MODE.upper() == "BOTH" or MODE.upper() == "TEST":
		testLoader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_embeddings,num_workers=NUMBER_OF_WORKERS)

		bestModel = torch.load(os.path.join(PATH_OUTPUT, f"{targetVariable}_GRU.pth"))
		_, _, testResults = test_model(bestModel, device, testLoader, criterion)

		y_true, y_pred = zip(*testResults)

		aurocScore = roc_auc_score(y_true, y_pred)
		auprcScore = average_precision_score(y_true, y_pred)
		f1Score = f1_score(y_true, y_pred)

		print('\nFinal Test scores: \n'
			  f'{targetVariable} AUROC Score: {aurocScore}\n'
			  f'{targetVariable} AUPRC Score: {auprcScore}\n'
			  f'{targetVariable} F1 Score: {f1Score}')

		plot_confusion_matrix(y_true, y_pred, ["No", "Yes"], f"{targetVariable.upper()} GRU")


def read_data():
	train_X = pd.read_pickle(f"{DATASET_FILE_PATH}/train_features.pkl")
	train_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/train_labels.pkl")

	validation_X = pd.read_pickle(f"{DATASET_FILE_PATH}/validation_features.pkl")
	validation_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/validation_labels.pkl")

	test_X = pd.read_pickle(f"{DATASET_FILE_PATH}/test_features.pkl")
	test_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/test_labels.pkl")

	patient_embed = pd.read_pickle(f"{DATASET_FILE_PATH}/patient_embeddings.pkl")
  

	return (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y), patient_embed


def dataframe_to_tensorDataset(train, validation, test, targetVariable,embeddings):

	patient_ids_with_embeddings = [id for id in embeddings.SUBJECT_ID]

	#print("# patients with ids:",len(patient_ids_with_embeddings))

	train_X, train_Y = train
	validation_X, validation_Y = validation
	test_X, test_Y = test

	# Only keepinf patients with embeddings
	train_X = train_X[train_X.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]
	train_Y = train_Y[train_Y.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]

	validation_X = validation_X[validation_X.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]
	validation_Y = validation_Y[validation_Y.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]

	test_X = test_X[test_X.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]
	test_Y = test_Y[test_Y.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]

	train_embed = embeddings[embeddings['SUBJECT_ID'].isin(train_X.index.get_level_values("subject_id"))]
	validation_embed = embeddings[embeddings['SUBJECT_ID'].isin(validation_X.index.get_level_values("subject_id"))]
	test_embed = embeddings[embeddings['SUBJECT_ID'].isin(test_X.index.get_level_values("subject_id"))]
	
	# data_train = torch.tensor(train_X.values, dtype=torch.float32).view(-1, 24, 104)
	#print("train shape:",data_train.shape)
	#print("train index=",data_train[0].shape)
	#print(len(set(train_X.index.get_level_values("subject_id"))))

	train = TimeSeriesDataset(seqs = torch.tensor(train_X.values, dtype=torch.float32).view(-1, 24, 104),
							  embeddings = train_embed['word2vec'].tolist(),
								labels = torch.tensor(train_Y[targetVariable].values))
	validation = TimeSeriesDataset(seqs = torch.tensor(validation_X.values, dtype=torch.float32).view(-1, 24, 104),
							  embeddings= validation_embed['word2vec'].tolist(),
								labels = torch.tensor(validation_Y[targetVariable].values))
	test = TimeSeriesDataset(seqs = torch.tensor(test_X.values, dtype=torch.float32).view(-1, 24, 104),
							  embeddings= test_embed['word2vec'].tolist(),
								labels = torch.tensor(test_Y[targetVariable].values))


	# train = TensorDataset(torch.tensor(train_X.values, dtype=torch.float32).view(-1, 24, 104),
	#                       torch.tensor(train_Y[targetVariable].values))
	# validation = TensorDataset(torch.tensor(validation_X.values, dtype=torch.float32).view(-1, 24, 104),
	#                            torch.tensor(validation_Y[targetVariable].values))
	# test = TensorDataset(torch.tensor(test_X.values, dtype=torch.float32).view(-1, 24, 104),
	#                      torch.tensor(test_Y[targetVariable].values))

	return train, validation, test


if __name__ == '__main__':
	trainSet, validationSet, testSet, embeddings = read_data()
	#targetVariable = "los_3"
	#newTrainSet, newValidationSet, newTestSet = dataframe_to_tensorDataset(trainSet, validationSet, testSet,
	#																		  targetVariable,embeddings)
	
	# trainLoader = DataLoader(newTrainSet, batch_size=32, shuffle=True,collate_fn=collate_embeddings,
	#  					   num_workers=NUMBER_OF_WORKERS)
	# batch = next(iter(trainLoader))
	# print(len(batch))
	# (embed,seqs),labels= batch
	# print(seqs.shape)
	# print(embed.shape)
	# print(labels.shape)
 

	for targetVariable in TARGET_VARIABLES:
		newTrainSet, newValidationSet, newTestSet = dataframe_to_tensorDataset(trainSet, validationSet, testSet,
																			   targetVariable,embeddings)
		
		model_runner(newTrainSet, newValidationSet, newTestSet, targetVariable)
