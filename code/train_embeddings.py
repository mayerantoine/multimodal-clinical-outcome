import os
import pandas as pd 
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import gensim


OUTPUT_MODEL_PATH = "../output/model/"
OUTPUT_PATH = "../output"
def train():
    patient_entities = pickle.load(open(f"{OUTPUT_PATH}/patient_entities.pkl", 'rb'))
    preprocess_notes = patient_entities['entities'].apply(lambda x:  [item[0] for item in x] if x else '' )
    model_w2vec = gensim.models.Word2Vec(window=10,min_count=2)
    model_w2vec.build_vocab(preprocess_notes,progress_per=10)

    print("Word2Vec Vocab size:",len(model_w2vec.wv))

    print("training word2vec.....")
    model_w2vec.train(preprocess_notes,total_examples=model_w2vec.corpus_count, epochs =model_w2vec.epochs)

    print("Save model...")
    model_w2vec.save(f'{OUTPUT_MODEL_PATH}/word2vec-mimiciii.model')

if __name__=='__main__':
    train()