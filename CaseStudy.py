# encoding: utf-8
import os.path

import pandas as pd
import torch
from Models.GraphSurv import GraphSurv
from Models.CoxNN import DeepCox_LossFunc
from Models.DeepSurv import NegativeLogLikelihood
from utils.Indicators import concordance_index
from utils.support import split_censor, split_data, sort_data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.EarlyStopping import EarlyStopping
from utils.DataProcessing import CancerDataset
import torch_geometric.transforms as T

import random

torch.set_printoptions(profile='full')

ROOT_DATA = "E:\\xiong\\Exp\\MyProject\\data\\cancer"
ROOT_TRAIN_MODEL = "E:\\xiong\\Exp\\MyProject\\SavedModels\\Train"
ROOT_PRETRAIN_MODEL = "E:\\xiong\\Exp\\MyProject\\SavedModels\\Pretrain"
ROOT_INDICATORS = "E:\\xiong\\Exp\\MyProject\\SavedIndicators"
writer = SummaryWriter(log_dir="E:\\xiong\\Exp\\MyProject\\log")

seed = 1234
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def save_evaluation_indicators(indicators, model_name):
    if not os.path.exists(ROOT_INDICATORS):
        os.makedirs(ROOT_INDICATORS)

    path = "{}\\{}_Indicators.xlsx".format(ROOT_INDICATORS, model_name)
    file = open(path, "a")
    """
    file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " +
               str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + " " +
               str(np.round(indicators[4], 4)) + "\n")
    """
    str_indicators = ''
    for index, indicator in enumerate(indicators):
        if index == 0:
            str_indicators += str(indicator) + " "
        else:
            str_indicators += str(np.round(indicator, 4)) + " "
    str_indicators += "\n"
    file.write(str_indicators)
    file.close()


def get_augmented_features(data, cvae_model, concat, device):
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
    return augmented_features


def run(cancer_name, data, lr):
    num_data = len(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_info = '{}\\{}\\processed\\{}_info.pkl'.format(ROOT_DATA, cancer_name, cancer_name)
    clinic_info = torch.load(path_info, map_location='cpu')
    event, duration = clinic_info['event'], clinic_info['duration']

    loader = DataLoader(dataset=data, batch_size=1, shuffle=False)
    num_nodes = data[0].num_nodes
    model_path = "{}\\{}".format(ROOT_TRAIN_MODEL, cancer_name)
    model = torch.load("{}\\{}_best_model.pth".format(model_path, cancer_name))
    cvae_model = torch.load('{}\\{}\\{}_pretrain.pth'.format(ROOT_PRETRAIN_MODEL, cancer_name, cancer_name)).to(
        device=device)

    Cindex, risks = evaluate(model=model, cvae_model=cvae_model, dataloader=loader,
                             test_time=duration, device=device)
    print(Cindex)

    median = risks.median()
    risk_group = []
    for index, item in enumerate(risks.numpy()):
        if item[0] > median:
            risk_group.append('high_risk')
        else:
            risk_group.append('low_risk')
    risk_df = pd.DataFrame(data=risk_group)
    risk_df.drop()
    risk_df.to_csv("risk_group.csv", index=False, header=False)

    pass


def evaluate(model, cvae_model, dataloader, test_time, device):
    model.eval()
    with torch.no_grad():
        risks = torch.zeros([len(dataloader), 1], dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            graphs = graphs.to(device)
            augmented_features = get_augmented_features(data=graphs, cvae_model=cvae_model, concat=1,
                                                        device=device)
            graphs.x = augmented_features + graphs.x
            risk, feature_gnn = model(graphs.x, graphs.adj_t)
            risks[id] = risk
        risks_save = risks.detach()
        cindex = concordance_index(test_time, -risks_save.cpu().numpy())
    return cindex, risks_save


cancer = 'BRCA'
data = CancerDataset(root=os.path.join(ROOT_DATA, cancer), transform=T.ToSparseTensor())
run(cancer_name=cancer, data=data, lr=None)
