import os
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data


def get_omics_data(features):
    duration = np.array(features.iloc[1, :].values.tolist()[2:])
    event = np.array(features.iloc[0, :].values.tolist()[2:])
    gene_exp = features.loc[features['Platform'] == 'geneExp']
    gene_exp = gene_exp.drop(columns=['Platform'], axis=1)

    copy_num = features.loc[features['Platform'] == 'copyNumber']
    copy_num = copy_num.drop(columns=['Platform'], axis=1)
    meth = features.loc[features['Platform'] == 'methylation']
    meth = meth.drop(columns=['Platform'], axis=1)

    # Merge omics data
    df = pd.merge(gene_exp, copy_num, on=['GeneSymbol'], how='inner')
    omic_data = pd.merge(df, meth, on=['GeneSymbol'], how='inner')

    return omic_data, duration, event


def get_edge(omic_data, gene_relationship):
    gene_name_omic = omic_data['GeneSymbol']
    gene_relationship.columns = ['gene_x', 'gene_y']
    gene_relationship = gene_relationship.drop_duplicates()
    gene_name_net = np.unique(gene_relationship.values)
    # Keep the genes in both omics data and KEGG pathway
    gene_name_comm = list(set(gene_name_net) & set(gene_name_omic))
    gene_idx_dic = {'gene_name': gene_name_comm, 'node_idx': list(np.arange(len(gene_name_comm)))}
    gene_idx = pd.DataFrame(gene_idx_dic)

    tmp1 = gene_idx.rename(columns={'gene_name': 'gene_x'})
    tmp_nodex = pd.merge(gene_relationship, tmp1, on='gene_x').drop_duplicates().reset_index(drop=True)

    tmp2 = gene_idx.rename(columns={'gene_name': 'gene_y'})
    adj_df = pd.merge(tmp_nodex, tmp2, on='gene_y').drop_duplicates().reset_index(drop=True)

    gene_name_final = np.unique(adj_df[['gene_x', 'gene_y']].values)
    gene_idx_dic = {'gene_name': gene_name_final, 'node_idx': list(np.arange(len(gene_name_final)))}
    gene_idx = pd.DataFrame(gene_idx_dic)
    tmp1 = gene_idx.rename(columns={'gene_name': 'gene_x'})
    tmp_nodex = pd.merge(gene_relationship, tmp1, on='gene_x').drop_duplicates().reset_index(drop=True)

    tmp2 = gene_idx.rename(columns={'gene_name': 'gene_y'})
    adj_df = pd.merge(tmp_nodex, tmp2, on='gene_y').drop_duplicates().reset_index(drop=True)
    adj_df = adj_df[['node_idx_x', 'node_idx_y']]
    return adj_df, gene_name_final


def read_data(raw_dir, raw_file_names, cancer):
    feature_path = os.path.join(raw_dir, raw_file_names[0])
    edge_path = 'E:\\xiong\\Exp\\MyProject\\data\\kegg\\kegg.csv'

    # Process features
    dataset_original = pd.read_csv(filepath_or_buffer=feature_path)
    gene_relationship = pd.read_csv(filepath_or_buffer=edge_path, sep=',', header=None)
    omic_data, time, status = get_omics_data(dataset_original)
    # Process edges
    edges, gene_name_final = get_edge(omic_data, gene_relationship)

    omic_data = omic_data.set_index(omic_data['GeneSymbol'].values)
    omic_data = omic_data.loc[gene_name_final]
    features = np.transpose(omic_data.drop(columns=['GeneSymbol'], axis=1).values)

    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_time = time.reshape(-1, 1)
    samples_num = features.shape[0] // 3
    if data_time.shape[0] == status.shape[0] == samples_num:
        print("There are %d samples" % (samples_num))
    gene_exp, copy_num, meth = features[0:samples_num], features[samples_num:2 * samples_num], features[
                                                                                               2 * samples_num:]

    return gene_exp, copy_num, meth, data_time, status, edges


def get_network_info():
    edge_path = 'E:\\xiong\\Exp\\MyProject\\data\\kegg\\kegg.csv'
    gene_relationship = pd.read_csv(filepath_or_buffer=edge_path, sep=',', header=None)
    gene_relationship = gene_relationship.drop_duplicates()
    num_interactions = gene_relationship.shape[0]
    num_nodes = len(set(gene_relationship[0]) | set(gene_relationship[1]))


ROOT_DATA = "E:\\xiong\\Exp\\MyProject\\data\\cancer"
cancers = os.listdir(ROOT_DATA)
for cancer in ['BRCA']:
    gene_exp, copy_num, meth, duration, event, edges = read_data(
        raw_dir='E:\\xiong\\Exp\\MyProject_Comparison\\data\\cancer\\{}\\raw'.format(cancer),
        raw_file_names=['{}.csv'.format(cancer)], cancer=cancer)
    print("cancer: {}, shape of gene_exp={}, shape of CNV={}, shape of METH={}, shape of edge={}".format(cancer,
                                                                                                         gene_exp.shape,
                                                                                                         copy_num.shape,
                                                                                                         meth.shape,
                                                                                                         edges.shape))
# get_network_info()
