import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time
from plots import plot_learning_curves, plot_confusion_matrix

MODE = "TRAIN"  # Options: 'BOTH', 'TRAIN', 'TEST'
TARGET_VARIABLE = "los_3"  # Options: 'mort_hosp', 'mort_icu', 'los_3', 'los_7'
NUMBER_OF_WORKERS = 0
EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.001

DATASET_FILE_PATH = "../output"
PATH_OUTPUT = "../output/models/"


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(104, 256, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.hiddenLayer = nn.Linear(256, 2)

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.sigmoid(output[:, -1, :])
        output = self.hiddenLayer(output)

        return output


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


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from utils.py
#
def calculate_accuracy(output, target):
    with torch.no_grad():
        batchSize = target.size(0)
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum()

        return correct * 100.0 / batchSize


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
        accuracy.update(calculate_accuracy(output, target).item(), target.size(0))

        print(f'Epoch: [{epoch}][{i}/{len(dataLoader)}]\t'
              f'Time {batchTime.value:.3f} ({batchTime.average:.3f})\t'
              f'Data {dataTime.value:.3f} ({dataTime.average:.3f})\t'
              f'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
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
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            batchTime.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(calculate_accuracy(output, target).item(), target.size(0))

            true = target.detach().cpu().numpy().tolist()
            predicted = output.detach().cpu().max(1)[1].numpy().tolist()
            results.extend(list(zip(true, predicted)))

            print(f'Test: [{i}/{len(dataLoader)}]\t'
                  f'Time {batchTime.value:.3f} ({batchTime.average:.3f})\t'
                  f'Loss {loss.val:.4f} ({loss.average:.4f})\t'
                  f'Accuracy {accuracy.value:.3f} ({accuracy.average:.3f})')

    return losses.average, accuracy.average, results


#
# Citation: Jimeng Sun, (2024). CSE6250: Big Data Analytics in Healthcare Homework 4
# Used code from train_seizure.py
#
def model_runner(train, validation, test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRU()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if MODE.upper() == "BOTH" or MODE.upper() == "TRAIN":
        trainLoader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS)
        validationLoader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUMBER_OF_WORKERS)

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
                torch.save(model, os.path.join(PATH_OUTPUT, "GRU.pth"))

        plot_learning_curves(trainLosses, validationLosses, trainAccuracies, validationAccuracies, f"{TARGET_VARIABLE.upper()} GRU")

    if MODE.upper() == "BOTH" or MODE.upper() == "TEST":
        testLoader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUMBER_OF_WORKERS)

        bestModel = torch.load(os.path.join(PATH_OUTPUT, "GRU.pth"))
        _, _, testResults = test_model(bestModel, device, testLoader, criterion)

        plot_confusion_matrix(testResults, ["No", "Yes"], f"{TARGET_VARIABLE.upper()} GRU")


def read_data():
    train_X = pd.read_pickle(f"{DATASET_FILE_PATH}/train_features.pkl")
    train_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/train_labels.pkl")

    validation_X = pd.read_pickle(f"{DATASET_FILE_PATH}/validation_features.pkl")
    validation_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/validation_labels.pkl")

    test_X = pd.read_pickle(f"{DATASET_FILE_PATH}/test_features.pkl")
    test_Y = pd.read_pickle(f"{DATASET_FILE_PATH}/test_labels.pkl")

    return (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y)


def dataframe_to_tensorDataset(train, validation, test):
    train_X, train_Y = train
    validation_X, validation_Y = validation
    test_X, test_Y = test

    train = TensorDataset(torch.tensor(train_X.values, dtype=torch.float32).view(-1, 24, 104),
                          torch.tensor(train_Y[TARGET_VARIABLE].values))
    validation = TensorDataset(torch.tensor(validation_X.values, dtype=torch.float32).view(-1, 24, 104),
                               torch.tensor(validation_Y[TARGET_VARIABLE].values))
    test = TensorDataset(torch.tensor(test_X.values, dtype=torch.float32).view(-1, 24, 104),
                         torch.tensor(test_Y[TARGET_VARIABLE].values))

    return train, validation, test


if __name__ == '__main__':
    trainSet, validationSet, testSet = read_data()

    trainSet, validationSet, testSet = dataframe_to_tensorDataset(trainSet, validationSet, testSet)

    model_runner(trainSet, validationSet, testSet)
