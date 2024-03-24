
import os
import pandas as pd 
import numpy as np
import pickle
from gensim.utils import simple_preprocess

DATASET_FILE_PATH = "../data/all_hourly_data.h5"
OUTPUT_PATH = "../output"
NOTES_FILEPATH  = '../data/notes/'



def _load_patient_ids():
    all_labels = pickle.load(open(f"{OUTPUT_PATH}/all_labels.pkl", 'rb'))
    patients = pd.read_hdf(DATASET_FILE_PATH, 'patients').reset_index()

    return all_labels, patients

def _load_clinical_notes():
    notes_files = [os.path.join(NOTES_FILEPATH,file) for file in os.listdir(os.path.join(NOTES_FILEPATH))]
    notes_df = pd.concat((pd.read_csv(f,delimiter=';') for f in notes_files), ignore_index=True)

    return notes_df

def _filter_clinical_notes(patients,notes_df,all_labels,preprocess=False):

    subjectIds = np.unique(np.array([subjectId for subjectId in all_labels.index.get_level_values("subject_id")]))

    notes_filtered_df = notes_df[notes_df['SUBJECT_ID'].isin(subjectIds)]
    patient_filtered_df = patients[patients['subject_id'].isin(subjectIds)]

    print("notes filtered shape:",notes_filtered_df.shape)

    patient_notes = patient_filtered_df.merge(notes_filtered_df[['SUBJECT_ID','HADM_ID','CHARTDATE', 'CHARTTIME','CATEGORY','TEXT']],
                                          right_on=['SUBJECT_ID'],
                                            left_on=['subject_id'],how='left')
    

    patient_notes = patient_notes[patient_notes['CATEGORY']!='Discharge summary']
    patient_notes = patient_notes[~patient_notes['CHARTTIME'].isna()]
    patient_notes['clinical_note_nday'] = ((pd.to_datetime(patient_notes['CHARTTIME'])-patient_notes['intime']).dt.days)
    patient_notes = patient_notes[patient_notes['clinical_note_nday'] <  1].reset_index()

   
    notes = patient_notes[['SUBJECT_ID','TEXT']].copy()
    if preprocess:
        notes = preproprocess_notes(notes)
    
    notes.to_pickle(f"{OUTPUT_PATH}/patient_notes.pkl")
    
    return notes

def preproprocess_notes(patient_notes):
    # TODO should we use same preprocessing
    # https://github.com/kaggarwal/ClinicalNotesICU
    # https://github.com/tanlab/ConvolutionMedicalNer/blob/master/preprocess.py

    patient_notes['TEXT'] = patient_notes['TEXT'].apply(lambda x: ' '.join(simple_preprocess(x)))

    return patient_notes

def main():

    print("loading patients..and labels..")
    all_labels,patients = _load_patient_ids()
    notes_df = _load_clinical_notes()

    print("filtering clincal notes...")
    patients_notes = _filter_clinical_notes(patients,notes_df,all_labels)


    print("MIMIC Extract =",patients.shape)
    print("Patients at least 30 hours =",all_labels.shape)
    print("All clinical notes=",notes_df.shape)
    print("Clinical notes filtered=",patients_notes.shape)
    print("Patients with clinical notes=",patients_notes['SUBJECT_ID'].nunique())

if __name__ =="__main__":
    main()