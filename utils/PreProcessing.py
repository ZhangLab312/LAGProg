import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def read_clinical(clinical_path, cancer):
    """
    TODO: Read clinic data
          0: ”Alive”
          1: ”Dead”
    :param path:
    :return:
    """
    clinical = pd.read_table(filepath_or_buffer=clinical_path, skiprows=[0, 2])

    clinical_new = clinical[["bcr_patient_barcode", "vital_status", "days_to_last_followup", "days_to_death"]]

    clinical_new["vital_status"] = (clinical_new["vital_status"] == 'Dead').astype(int)

    clinical_new["days_to_last_followup"][clinical_new["days_to_last_followup"] == "[Not Available]"] = 0
    clinical_new["days_to_last_followup"][clinical_new["days_to_last_followup"] == "[Discrepancy]"] = 0
    clinical_new["days_to_last_followup"][clinical_new["days_to_last_followup"] == "[Completed]"] = 0
    clinical_new["days_to_death"][clinical_new["days_to_death"] == "[Not Available]"] = 0
    clinical_new["days_to_death"][clinical_new["days_to_death"] == "[Not Applicable]"] = 0
    clinical_new["days_to_death"][clinical_new["days_to_death"] == "[Discrepancy]"] = 0
    clinical_new["days_to_death"][clinical_new["days_to_death"] == "[Completed]"] = 0
    clinical_new["days_to_last_followup"] = pd.to_numeric(clinical_new["days_to_last_followup"])
    clinical_new["days_to_death"] = pd.to_numeric(clinical_new["days_to_death"])
    clinical_new["days_to_last_followup"] = clinical_new["days_to_last_followup"] + clinical_new["days_to_death"]
    clinical_new = clinical_new.drop(columns="days_to_death")
    clinical_new.columns = ["barcode", "status", "time"]

    save_path = os.path.dirname(clinical_path)
    clinical_new.to_csv(os.path.join(save_path, "{}_Clinic.csv".format(cancer)), index=False)
    return clinical_new


def process_duplicates(feature_path):
    """


    :param feature_path: {cancer}_Combined.txt
    :return:
    """
    features = pd.read_table(feature_path, sep='\t')

    features['GeneSymbol'] = features['GeneSymbol'] + "_" + features['Platform']

    features = features.drop(columns=["Platform", "Description"])

    duplicated_features = features[features.duplicated(subset='GeneSymbol')]
    duplicated_gene = duplicated_features['GeneSymbol'].values
    for gene in duplicated_gene:
        temp = features.loc[features['GeneSymbol'] == gene]
        temp_mean = temp.iloc[:, 1:].mean().values
        index = temp.index.values

        features.iloc[index, 1:] = temp_mean

    features.drop_duplicates(keep='first', inplace=True)
    return features


def process_missing_value(feature_path):
    """

    TODO: Process missing value:
    :param features:
    :return:
    """

    features = process_duplicates(feature_path=feature_path)

    features = features.fillna(0)

    missing_sample = (features == 0).astype(int).sum(axis=0) / features.shape[0]
    features = features.loc[:, missing_sample < 0.2]

    missing_feature = (features == 0).astype(int).sum(axis=1) / (features.shape[1] - 1)
    features = features.loc[missing_feature < 0.2, :]
    features = features.reset_index(drop=True)
    return features


def nomalization(features, cancer):
    """
    TODO: Z-score normalization
    :param features:
    :return:
    """
    root_path = ""
    features_arr = features.values

    for index, item in enumerate(features_arr):

        feature_name = item[0]
        if 'geneExp' in feature_name or 'miRNAExp' in feature_name:
            features.iloc[index, 1:] = np.log2(np.array(features.iloc[index, 1:].values, dtype=float) + 1)

        features_value = features.iloc[index, 1:].values
        features.iloc[index, 1:] = (features_value - features_value.mean()) / features_value.std()

    features.to_csv("{}\\{}\\{}_norm.csv".format(DATA_ROOT, cancer, cancer), index=False)


def combine_feature_clinic(feature_path, clinic_path, cancer):
    features = pd.read_csv(feature_path)
    clinic = pd.read_csv(clinic_path)

    sample_barcodes = features.columns.values[1:]
    clinic_barcodes = clinic['barcode'].values

    clinic_list = [['Event_n'], ['Duration_n']]
    for i in range(len(sample_barcodes)):

        split_barcode = sample_barcodes[i].split('-')
        new_sample_barcode = "{}-{}-{}".format(split_barcode[0], split_barcode[1], split_barcode[2])

        if new_sample_barcode in clinic_barcodes:
            clinic_list[0].append(clinic[clinic['barcode'] == new_sample_barcode]['status'].values[0])
            clinic_list[1].append(abs(clinic[clinic['barcode'] == new_sample_barcode]['time'].values[0]))
        else:

            features = features.drop(columns=[sample_barcodes[i]])

    clinic_df = pd.DataFrame(data=clinic_list, columns=features.columns)

    feature_combined = pd.concat([clinic_df, features], ignore_index=True)
    feature_combined['GeneSymbol_new'] = feature_combined['GeneSymbol'].map(lambda x: x.split('_')[0])
    feature_combined['platform'] = feature_combined['GeneSymbol'].map(lambda x: x.split('_')[1])
    feature_combined['GeneSymbol'] = feature_combined['GeneSymbol_new']
    feature_combined.insert(loc=1, column='Platform', value=feature_combined['platform'])
    feature_combined = feature_combined.drop(columns=['GeneSymbol_new', 'platform'], axis=1)
    feature_combined.to_csv("{}\\{}\\{}.csv".format(DATA_ROOT, cancer, cancer), index=False)


DATA_ROOT = "E:\\xiong\\Bioinfomatics\\Disease\\Data"
cancers = os.listdir(DATA_ROOT)
cancer_bar = tqdm(cancers)
for cancer in cancer_bar:
    cancer_bar.set_description("{}".format(cancer))

    if not os.path.exists("{}\\{}\\{}_Clinic.csv".format(DATA_ROOT, cancer, cancer)):
        _ = read_clinical("{}\\{}\\{}_Clinic.txt".format(DATA_ROOT, cancer, cancer), cancer)
    if not os.path.exists("{}\\{}\\{}_norm.csv".format(DATA_ROOT, cancer, cancer)):
        features = process_missing_value("{}\\{}\\{}_Combined.txt".format(DATA_ROOT, cancer, cancer))
        nomalization(features=features, cancer=cancer)
    if not os.path.exists("{}\\{}\\{}.csv".format(DATA_ROOT, cancer, cancer)):
        combine_feature_clinic(feature_path='{}\\{}\\{}_norm.csv'.format(DATA_ROOT, cancer, cancer),
                               clinic_path='{}\\{}\\{}_Clinic.csv'.format(DATA_ROOT, cancer, cancer), cancer=cancer)
