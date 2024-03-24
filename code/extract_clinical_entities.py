import os
import pandas as pd 
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import progressbar

from gensim.utils import simple_preprocess

OUTPUT_PATH = "../output"

med7 = spacy.load("en_core_med7_lg")
def extract_entities(text):
    #text = text.replace("_","")
    #text = text.strip()
    #text = text.replace("\n",'')
    doc = med7(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if len(entities)< 1:
        return None
    else:
        return entities


def generate_notes_entities(patient_notes):

    print("total notes=",len(patient_notes))
    print("total patients=",patient_notes['SUBJECT_ID'].nunique())
    #vfunc = np.vectorize(extract_entities)
    #patient_notes['entities'] = [extract_entities(text.TEXT) for text in patient_notes.itertuples()]
    #entipatient_notes['entities'] = list(map(extract_entities,patient_notes['TEXT']))
    #patient_notes['entities'] = vfunc(patient_notes['TEXT'])

    text_ner = []
    i = 0
    with progressbar.ProgressBar(max_value=len(patient_notes)) as bar:
        for text in patient_notes.itertuples():
            entities = extract_entities(text.TEXT)
            text_ner.append(entities)
            bar.update(i)
            i = i + 1

    patient_notes['entities'] = text_ner
    pickle.dump(patient_notes,open(f"{OUTPUT_PATH}/patient_entities.pkl",'wb'),pickle.HIGHEST_PROTOCOL)

    return patient_notes


def _count_entities(patient_entities):

    med7_data = {'DRUG':[],'DOSAGE':[],'FORM':[],'ROUTE':[],'STRENGTH':[],'FREQUENCY':[],'DURATION':[]}

    for entities in patient_entities['entities']:
        if entities:
            for item in entities:
                med7_data[item[1]].append(item[0])
    
    return med7_data


def main():
    patient_notes = pickle.load(open(f"{OUTPUT_PATH}/patient_notes.pkl", 'rb'))

    print("Extracting clinical entities using med7.....")
    patient_entities = generate_notes_entities(patient_notes)

    med7_data = _count_entities(patient_entities)
    total_count = { k: len(v) for k,v in med7_data.items()}
    total_unique = { k: len(set(v)) for k,v in med7_data.items()}
                
    df_total = pd.DataFrame.from_dict(total_count,orient='index')#.reset_index()
    df_total.columns=['total_count']
    df_total['unique_count'] = pd.DataFrame.from_dict(total_unique,orient='index')
    print(df_total)


if __name__=='__main__':
    main()