import pandas as pd
import numpy as np

GAP_TIME = 6
WINDOW_SIZE = 24
ID_COLUMNS = ["subject_id", "hadm_id", "icustay_id"]

DATASET_FILE_PATH = "../data/all_hourly_data.h5"
OUTPUT_PATH = "../output"


def read_data():
    df_patients = pd.read_hdf(DATASET_FILE_PATH, "patients")

    df_vitals_labs = pd.read_hdf(DATASET_FILE_PATH, "vitals_labs")

    return df_patients, df_vitals_labs


def filter_data(df_patients, df_vitals_labs):
    df_patients = df_patients[df_patients["max_hours"] > WINDOW_SIZE + GAP_TIME][["mort_hosp", "mort_icu", "los_icu"]]

    df_vitals_labs = df_vitals_labs[
        (df_vitals_labs.index.get_level_values("icustay_id").isin(df_patients.index.get_level_values("icustay_id"))) & (
                df_vitals_labs.index.get_level_values("hours_in") < WINDOW_SIZE)]

    return df_patients, df_vitals_labs


def clean_data(df_vitals_labs):
    if len(df_vitals_labs.columns.names) > 2:
        df_vitals_labs.columns = df_vitals_labs.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))

    df_vitals_labs = df_vitals_labs.xs("mean", level=1, axis=1)

    df_vitals_labs_means = df_vitals_labs.groupby(ID_COLUMNS).mean()

    df_vitals_labs = df_vitals_labs.groupby(ID_COLUMNS).fillna(method='ffill').fillna(df_vitals_labs_means).fillna(0)

    return df_vitals_labs


def create_labels(df_patients):
    df_labels = df_patients[["los_icu", "mort_hosp", "mort_icu"]].copy()

    df_labels["los_3"] = (df_labels["los_icu"] > 3).astype(int)
    df_labels["los_7"] = (df_labels["los_icu"] > 7).astype(int)

    df_labels.drop(columns=["los_icu"], inplace=True)

    return df_labels


def split_data(df_features, df_labels, trainPercentage, validationPercentage):
    subjectIds = np.unique(np.array([subjectId for subjectId in df_labels.index.get_level_values("subject_id")]))

    np.random.shuffle(subjectIds)

    subjectIdsCount = len(subjectIds)
    trainEnd = int(subjectIdsCount * trainPercentage)
    validationEnd = int((trainPercentage + validationPercentage) * subjectIdsCount)

    trainSubjectIds = subjectIds[:trainEnd]
    validationSubjectIds = subjectIds[trainEnd:validationEnd]
    testSubjectIds = subjectIds[validationEnd:]

    def create_set(ids):
        X = df_features[df_features.index.get_level_values("subject_id").isin(ids)]
        Y = df_labels[df_labels.index.get_level_values("subject_id").isin(ids)]

        return X, Y

    return create_set(trainSubjectIds), create_set(validationSubjectIds), create_set(testSubjectIds)


def output_to_file(train, validation, test):
    train[0].to_pickle(f"{OUTPUT_PATH}/train_features.pkl")
    train[1].to_pickle(f"{OUTPUT_PATH}/train_labels.pkl")

    validation[0].to_pickle(f"{OUTPUT_PATH}/validation_features.pkl")
    validation[1].to_pickle(f"{OUTPUT_PATH}/validation_labels.pkl")

    test[0].to_pickle(f"{OUTPUT_PATH}/test_features.pkl")
    test[1].to_pickle(f"{OUTPUT_PATH}/test_labels.pkl")


if __name__ == '__main__':
    patients, vitals_labs = read_data()

    patients, vitals_labs = filter_data(patients, vitals_labs)

    vitals_labs = clean_data(vitals_labs)

    labels = create_labels(patients)

    trainSet, validationSet, testSet = split_data(vitals_labs, labels, 0.7, 0.1)

    output_to_file(trainSet, validationSet, testSet)
