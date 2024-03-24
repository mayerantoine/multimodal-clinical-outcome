import pickle
import torch
import utils
#from utils import collate_embeddings
from torch.utils.data import DataLoader
from models import ConvolutionNER

OUTPUT_PATH = "../output"
OUTPUT_MODEL_PATH = "../output/model/"

EPOCHS = 10
BATCH_SIZE = 8

## TODO we might switch to pytorch lightning - less verbose, less code , more features
def train(data_loader,model,criterion,optimizer):
    model.train()
    for i, (input, target) in enumerate(data_loader):
        
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(loss.item())
    



def main():

    ## TODO argparse argument for LABEL mort_hosp, mmort_ics, los3, los7
    # Data loading
    print('===> Loading entire datasets')
    train_seqs = pickle.load(open(f"{OUTPUT_PATH}/train_features.pkl", 'rb'))
    train_labels = pickle.load(open(f"{OUTPUT_PATH}/train_labels.pkl", 'rb'))
    valid_seqs = pickle.load(open(f"{OUTPUT_PATH}/validation_features.pkl", 'rb'))
    valid_labels = pickle.load(open(f"{OUTPUT_PATH}/validation_labels.pkl", 'rb'))
    test_seqs = pickle.load(open(f"{OUTPUT_PATH}/test_features.pkl", 'rb'))
    test_labels = pickle.load(open(f"{OUTPUT_PATH}/test_labels.pkl", 'rb'))


    ## TODO We need to select patient with embeddings from train, valid and test
    
    patient_embed = pickle.load(open(f"{OUTPUT_PATH}/patient_embeddings.pkl", 'rb'))
    all_labels = pickle.load(open(f"{OUTPUT_PATH}/all_labels.pkl", 'rb'))

    patient_ids_with_embeddings = [id for id in patient_embed.SUBJECT_ID]

    labels = all_labels[all_labels.index.get_level_values("subject_id").isin(patient_ids_with_embeddings)]
    labels_mort_hosp = labels["mort_hosp"] .tolist()
    seqs = patient_embed['feature_embeddings'].tolist()

    train_dataset = utils.EmbeddingsDataset(seqs,labels_mort_hosp)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,collate_fn=utils.collate_embeddings)


    ## TODO this is only the Proposed model branch
    model_conv = ConvolutionNER(dim_input=100)  
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_conv.parameters(),lr=1e-3)

    for epoch in range(EPOCHS):
        print(f"Epochs={epoch}")
        train(train_loader,model_conv,criterion,optimizer)



if __name__ =="__main__":
    main()