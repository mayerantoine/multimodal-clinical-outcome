import os
import pandas as pd 
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import progressbar
import gensim


OUTPUT_PATH = "../output"
OUTPUT_MODEL_PATH = "../output/model/"

def _get_embeddings(model,enities_seq):
    feature_seq = []
    for ent in enities_seq:
        try:
            ent_embedding = model.wv[ent]
            feature_seq.append(ent_embedding.tolist())
        except:
            #print("Failed to get embeddings")
            pass
    return np.asarray(feature_seq)


def flatten_comprehension(matrix):     
    rs = []
    for row in matrix:
        if row:
            for item in row:
                rs.append(item)
    return list(set(rs))

def main():
    patient_entities = pickle.load(open(f"{OUTPUT_PATH}/patient_entities.pkl", 'rb'))
    patient_notes = pickle.load(open(f"{OUTPUT_PATH}/patient_notes.pkl", 'rb'))
    model_word2vec = gensim.models.Word2Vec.load(f'{OUTPUT_MODEL_PATH}/word2vec-mimiciii.model')

    # get a list of entities only
    patient_entities['entities_seq'] = patient_entities['entities'].apply(lambda x:  [item[0] for item in x] if x else None )

    #group by patient - for each patient a list of entities
    df1_seq = patient_entities[['SUBJECT_ID','entities_seq']].groupby('SUBJECT_ID').agg(list).reset_index()

    # flatten and create a set
    df1_seq['flat_entities_seq'] = df1_seq['entities_seq'].apply(lambda x: flatten_comprehension(x))

    # get embeddings from wor2vec
    df1_seq['feature_embeddings']= df1_seq['flat_entities_seq'].apply(lambda x :_get_embeddings(model_word2vec,x))

    # patients ids with no embeddings extracted from text
    ids_no_emb = [row.SUBJECT_ID for row in df1_seq.itertuples() if len(row.feature_embeddings)<1]

    print(df1_seq)
    print("patients ids with no embeddings",ids_no_emb)

    # remove patients with no embeddings
    df2_seq = df1_seq.set_index('SUBJECT_ID').drop(index=ids_no_emb).reset_index()

    pickle.dump(df2_seq[['SUBJECT_ID','feature_embeddings']],open(f"{OUTPUT_PATH}/patient_embeddings.pkl",'wb'),pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()