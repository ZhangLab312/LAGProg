import os
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Callable

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
    # Keep the genes in both omics data and KEGG pathway
    df = pd.merge(gene_exp, copy_num, on=['GeneSymbol'], how='inner')
    omic_data = pd.merge(df, meth, on=['GeneSymbol'], how='inner')

    return omic_data, duration, event


def get_edge(omic_data, gene_relationship):
    gene_name = omic_data['GeneSymbol']
    gene_relationship.columns = ['gene_x', 'gene_y']

    gene_idx_dic = {'gene_name': gene_name, 'node_idx': list(np.arange(len(gene_name)))}
    gene_idx = pd.DataFrame(gene_idx_dic)

    tmp1 = gene_idx.rename(columns={'gene_name': 'gene_x'})
    tmp_nodex = pd.merge(gene_relationship, tmp1, on='gene_x').drop_duplicates().reset_index(drop=True)

    tmp2 = gene_idx.rename(columns={'gene_name': 'gene_y'})
    adj_df = pd.merge(tmp_nodex, tmp2, on='gene_y').drop_duplicates().reset_index(drop=True)

    adj_df = adj_df[['node_idx_x', 'node_idx_y']]

    return adj_df


def read_data(raw_dir, raw_file_names):
    feature_path = os.path.join(raw_dir, raw_file_names[0])
    edge_path = 'E:\\xiong\\Exp\\MyProject\\data\\kegg\\kegg.csv'
    # Process features
    dataset_original = pd.read_csv(filepath_or_buffer=feature_path)
    gene_relationship = pd.read_csv(filepath_or_buffer=edge_path, sep=',', header=None)
    omic_data, time, status = get_omics_data(dataset_original)
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

    # Process edges
    edges = get_edge(omic_data, gene_relationship)
    return gene_exp, copy_num, meth, data_time, status, edges


class CancerDataset(InMemoryDataset):

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        names = os.listdir(self.raw_dir)
        return names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        names = ['{}.pt'.format(os.path.basename(self.root))]
        return names

    def download(self):
        pass

    def save_clinic(self, duration, event):
        torch.save(obj={'duration': duration, 'event': event},
                   f=os.path.join(self.processed_dir, '{}_info.pkl'.format(os.path.basename(self.root))))

    def process(self):
        data_list = []
        gene_exp, copy_num, meth, duration, event, edges = read_data(raw_dir=self.raw_dir,
                                                                     raw_file_names=self.raw_file_names)
        edge_index = torch.stack([torch.tensor(edges['node_idx_x'].to_list(), dtype=torch.long),
                                  torch.tensor(edges['node_idx_y'].to_list(), dtype=torch.long)])

        print('transforming samples to pyg graphs')
        for index, gene in enumerate(gene_exp):
            feature = np.vstack((gene, copy_num[index]))
            feature = np.vstack((feature, meth[index]))
            feature = torch.tensor(data=feature, dtype=torch.float32).t()
            data_list.append(Data(x=feature, edge_index=edge_index))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list=data_list)
        torch.save((data, slices), self.processed_paths[0])
        self.save_clinic(duration=duration, event=event)
