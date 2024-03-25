import os
import pandas as pd 
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import progressbar
import gensim
import argparse


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

def _get_contact_embeddings(fasttext,word2vec,entities_seq):
    feature_seq = []
    pad = np.zeros(100)
    for ent in entities_seq:

        try:
            ent_word2vec = word2vec.wv[ent]
        except:
            #print("Failed to get embeddings")
            ent_word2vec = None
        
        try:
            ent_fasttext = fasttext.wv[ent]
        except:
            ent_fasttext = None
        
        if (ent_word2vec is not None) and (ent_fasttext is not None):
            embed = np.concatenate([ent_word2vec,ent_fasttext])
            
        else:
            if (ent_word2vec is not None) and (ent_fasttext is None):
                embed = np.concatenate([ent_word2vec,pad])
                #print("ent no fasttext:",ent)
            if (ent_fasttext is not None) and (ent_word2vec is None):
                embed = np.concatenate([pad,ent_fasttext])
                #print("ent no word2vec:",ent)
        
        feature_seq.append(embed.tolist())
    
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
    #patient_notes = pickle.load(open(f"{OUTPUT_PATH}/patient_notes.pkl", 'rb'))

    model_word2vec = gensim.models.Word2Vec.load(f'{OUTPUT_MODEL_PATH}/word2vec-mimiciii.model')
    model_fasttext = gensim.models.FastText.load(f'{OUTPUT_MODEL_PATH}/fasttext-mimiciii.model')


    # get a list of entities only
    patient_entities['entities_seq'] = patient_entities['entities'].apply(lambda x:  [item[0] for item in x] if x else None )

    #group by patient - for each patient a list of entities
    df1_seq = patient_entities[['SUBJECT_ID','entities_seq']].groupby('SUBJECT_ID').agg(list).reset_index()

    # flatten and create a set
    df1_seq['flat_entities_seq'] = df1_seq['entities_seq'].apply(lambda x: flatten_comprehension(x))

    # get embeddings from wor2vec
    df1_seq['word2vec']= df1_seq['flat_entities_seq'].apply(lambda x :_get_embeddings(model_word2vec,x))
    df1_seq['fasttext']= df1_seq['flat_entities_seq'].apply(lambda x :_get_embeddings(model_fasttext,x))
    df1_seq['concat']= df1_seq['flat_entities_seq'].apply(lambda x :_get_contact_embeddings(model_fasttext,model_word2vec,x))

    # patients ids with no embeddings extracted from text
    ids_no_emb_word2vec = [row.SUBJECT_ID for row in df1_seq.itertuples() if len(row.word2vec)<1]
    ids_no_emb_fasttext = [row.SUBJECT_ID for row in df1_seq.itertuples() if len(row.fasttext)<1]
    ids_no_emb_concat= [row.SUBJECT_ID for row in df1_seq.itertuples() if len(row.concat)<1]
    
    print("patients ids with no word2vec",len(ids_no_emb_word2vec))
    print("patients ids with no fasttext",len(ids_no_emb_fasttext))
    print("patients ids with no concat",len(ids_no_emb_concat))

    # remove patients with no embeddings
    df2_seq = df1_seq.set_index('SUBJECT_ID').drop(index=ids_no_emb_word2vec).reset_index()


    pickle.dump(df2_seq[['SUBJECT_ID','word2vec','fasttext','concat']],open(f"{OUTPUT_PATH}/patient_embeddings.pkl",'wb'),pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()