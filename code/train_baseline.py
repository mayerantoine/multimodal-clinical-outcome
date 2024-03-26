import pickle
import torch
import utils
import numpy as np
from utils import TimeSeriesDataset
from torch.utils.data import DataLoader
from models import ConvolutionNERLightning,BaselineTimeSeriesGRU
import pytorch_lightning as pl

OUTPUT_PATH = "../output"
OUTPUT_MODEL_PATH = "../output/model/"

EPOCHS = 1
BATCH_SIZE = 8

def create_timeseries_seqs(data_timeseries):
    df = data_timeseries.reset_index()
    df = df.drop(['hadm_id', 'icustay_id', 'hours_in',],axis=1).set_index('subject_id')
    df['feature']= df.apply(lambda x: x.values.tolist(),axis=1)
    df = df.reset_index()
    df_feature = df[['subject_id','feature']].groupby('subject_id').agg(list).reset_index()

    return df_feature['feature'].tolist()

def main():

    ## TODO argparse argument for LABEL mort_hosp, mmort_ics, los3, los7
    # Data loading
    print('===> Loading entire datasets')
    train_features = pickle.load(open(f"{OUTPUT_PATH}/train_features.pkl", 'rb'))
    train_labels = pickle.load(open(f"{OUTPUT_PATH}/train_labels.pkl", 'rb'))
    validation_features = pickle.load(open(f"{OUTPUT_PATH}/validation_features.pkl", 'rb'))
    valid_labels = pickle.load(open(f"{OUTPUT_PATH}/validation_labels.pkl", 'rb'))
    test_features = pickle.load(open(f"{OUTPUT_PATH}/test_features.pkl", 'rb'))
    test_labels = pickle.load(open(f"{OUTPUT_PATH}/test_labels.pkl", 'rb'))

    train_seqs = create_timeseries_seqs(train_features)
    validation_seqs = create_timeseries_seqs(validation_features)
    #test_seqs = create_timeseries_seqs(test_features)

    print("total patient = ",len(train_labels)+len(valid_labels)+len(test_labels))

    train_labels_mort_hosp = train_labels["mort_hosp"].tolist()
    valid_labels_mort_hosp = valid_labels["mort_hosp"].tolist()

    print("train =",np.asarray(train_seqs).shape)
    print("validation=",np.asarray(validation_seqs).shape)
    #print("test=",np.asarray(test_seqs).shape)

    train_dataset = TimeSeriesDataset(train_seqs,train_labels_mort_hosp)
    validation_dateset = TimeSeriesDataset(validation_seqs,valid_labels_mort_hosp)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    validation_loader = DataLoader(dataset=validation_dateset, batch_size=BATCH_SIZE,shuffle=False)
    
    model = BaselineTimeSeriesGRU(dim_input=104)
    trainer= pl.Trainer(max_epochs=EPOCHS)

    trainer.fit(model,train_loader,validation_loader)
    trainer.validate(model,validation_loader)

    



if __name__== '__main__':
    main()